import torch
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Solver():
    """Solver for training and testing a model with parameters defined by the user in the config file."""
    def __init__(self, train_loader, test_loader, device, model, writer, args):
        self.args = args

        self.model_path=os.path.join(self.args.checkpoint_path,self.args.run_name)

        self.epochs = args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Patience for early stopping, ask the user to input it
        self.patience=5
        self.model = model
        
        self.device = device
        self.model.to(self.device)


        self.resume_epoch=0
        self.resume_iteration=-1
        if args.resume_train:
            self.resume_epoch,self.resume_iteration=self.load_model()
            print("Model loaded! Resuming training from epoch: ",self.resume_epoch," iteration: ",self.resume_iteration)


        if args.criterion == "BCELoss":
            self.criterion = torch.nn.BCELoss()
        elif args.criterion == "MSELoss":
            self.criterion = torch.nn.MSELoss()

        if args.opt == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,weight_decay=args.weight_decay)


        self.writer = writer



    def save_model(self, epoch,iteration):
        # if you want to save the model

        os.makedirs(self.model_path, exist_ok=True)

        
        # save the model with the name of the class and the epoch and iteration
        path = os.path.join(self.model_path, 
                                    self.model.__class__.__name__ +"_"+str(epoch)+"_"+str(iteration)+".pth")

        torch.save(self.model.state_dict(), path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        if not os.path.exists(self.model_path):
            raise "No checkpoint found!"

        if self.model_path[-1] == '/':
            self.model_path = self.model_path
        
        saved_models= glob.glob(self.model_path + '/*.pth')

        if len(saved_models) == 0:
            raise "No model found!"

        # get the last model saved in the folder
        last_model= max(saved_models, key=os.path.getctime)

        epoch, iteration = last_model.split("_")[-2:]
        iteration=iteration.split(".")[0]

        self.model.load_state_dict(torch.load(last_model))
        print("Model loaded!")

        return int(epoch), int(iteration)


    def train(self):
        print("Training...")

        evalutation_best_loss = 100000
        evalutation_best_accuracy= -1

        patience_counter=0

        for epoch in range(self.resume_epoch,self.epochs):
            self.model.train()
            running_loss = 0.0
            accuracy = 0.0

            for i, (x, y) in enumerate(self.train_loader,0):

                # if we are resuming the training we skip the iterations that we already did
                if (i <= self.resume_iteration and epoch == self.resume_epoch):
                    print("Skipping iteration: ",i," epoch: ",epoch)
                    continue

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.criterion(output, y)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                accuracy += (torch.gt(output,0.5).int() == y).sum().item() / y.shape[0]


                if i % self.args.print_every == self.args.print_every-1:  # print statistics, average loss over the last print_every mini-batches
                   
                    self.writer.add_scalar("Loss/train",running_loss / self.args.print_every, epoch * len(self.train_loader) + i)
                    self.writer.add_scalar("Accuracy/train",accuracy / self.args.print_every, epoch * len(self.train_loader) + i)

                    # self.writer('Gradients',
                    #         add_gradient_hist(self.model),
                    #         global_step=epoch * len(trainloader) + i))

                    print("Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}".format(epoch, i, running_loss / self.args.print_every, accuracy / self.args.print_every))
                    running_loss = 0.0
                    accuracy = 0.0
                    
                    self.save_model(epoch, i)

                self.writer.flush()

            # Early stopping 
            evalutation_loss, evaluation_accuracy= self.evaluate()


            self.writer.add_scalar("Loss/eval",evalutation_loss,epoch)
            self.writer.add_scalar("Accuracy/eval",evaluation_accuracy,epoch)

            # For early stopping we use the loss on the validation set not the accuracy
            if evalutation_loss < evalutation_best_loss:
                evalutation_best_loss = evalutation_loss
                evalutation_best_accuracy = evaluation_accuracy

                patience_counter=0
                # Save only the best model
                self.save_model(epoch, len(self.train_loader))
            # Accept a 5% increase in the loss
            elif evalutation_loss > evalutation_best_loss*1.05:
                patience_counter+=1
                # If we have reached the patience we stop the training
                if patience_counter == self.patience:
                    break
                print("The model did not improve and has started to overfit, the best performances are with loss of: {}  and accuracy of: {}".format(evalutation_best_loss, evalutation_best_accuracy))
            else:
                break
        
        # Load the best model
        self.load_model()
        self.test()

    def add_gradient_hist(net):
        ave_grads = [] 
        layers = []
        for n,p in net.named_parameters():
            if ("bias" not in n):
                layers.append(n)
                if p.requires_grad: 
                    ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
                else:
                    ave_grad = 0
                ave_grads.append(ave_grad)

        layers = [layers[i].replace(".weight", "") for i in range(len(layers))]

        fig = plt.figure(figsize=(12, 12))
        plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        #plt.grid(True)
        plt.legend([Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
        plt.tight_layout()
        #plt.show()

        return fig

    def evaluate(self, write_to_tensorboard=False):
        print("Testing...")
        self.model.eval()

        running_loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader,0):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.criterion(output, y)

                running_loss+= loss.item()
                accuracy += (torch.gt(output,0.5).int() == y).sum().item() / y.shape[0]

        
        print("Test loss: {}, Test accuracy: {}".format(running_loss / len(self.test_loader), accuracy / len(self.test_loader)))

        if write_to_tensorboard:
            self.writer.add_text("Test loss", running_loss / len(self.test_loader))
            self.writer.add_text("Test accuracy", accuracy / len(self.test_loader))

        return running_loss / len(self.test_loader), accuracy / len(self.test_loader)

            
