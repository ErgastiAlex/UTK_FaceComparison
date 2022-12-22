import torch
import os
import glob

class Solver():
    def __init__(self, train_loader, test_loader, device, model, writer, args):
        self.args = args

        self.epochs = args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = model
        
        self.device = device
        self.model.to(self.device)


        if args.resume_train:
            self.load_model()


        if args.criterion == "BCELoss":
            self.criterion = torch.nn.BCELoss()
        elif args.criterion == "MSELoss":
            self.criterion = torch.nn.MSELoss()

        if args.opt == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)


        self.writer = writer



    def save_model(self, epoch,iteration):
        # if you want to save the model
        if not os.path.exists(self.args.checkpoint_path):
            os.mkdir(self.args.checkpoint_path)
        
        # save the model with the name of the class and the epoch and iteration
        path = os.path.join(self.args.checkpoint_path, 
                                    self.model.__class__.__name__ +"_"+str(epoch)+"_"+str(iteration)+".pth")

        torch.save(self.model.state_dict(), path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        if not os.path.exists(self.args.checkpoint_path):
            raise "No checkpoint found!"

        if self.args.checkpoint_path[-1] == '/':
            self.args.checkpoint_path = self.args.checkpoint_path[:-1]
        
        saved_models= glob.glob(self.args.checkpoint_path + '/*.pth')

        if len(saved_models) == 0:
            raise "No model found!"

        # get the last model saved in the folder
        last_model= max(saved_models, key=os.path.getctime)

        self.model.load_state_dict(torch.load(last_model))
        print("Model loaded!")


    def train(self):
        print("Training...")

        running_loss = 0.0
        accuracy = 0.0

        evalutation_best_performance = 100000

        for epoch in range(self.epochs):
            self.model.train()

            for i, (x, y) in enumerate(self.train_loader,0):
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.criterion(output, y)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                accuracy += (torch.gt(output).int() == y).sum().item() / y.shape[0]


                #TODO add PR curve

                if i % self.args.print_every == self.args.print_every-1:  # print statistics, average loss over the last print_every mini-batches
                   
                    self.writer.add_scalar("Loss/train",running_loss / self.args.print_every, epoch * len(self.train_loader) + i)
                    self.writer.add_scalar("Accuracy/train",accuracy / self.args.print_every, epoch * len(self.train_loader) + i)

                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, running_loss / self.args.print_every))
                    running_loss = 0.0
                    
                    self.save_model(epoch, i)

                self.writer.flush()

            # Early stopping 
            evalutation_loss= self.evaluate()

            self.writer.add_scalar("Loss/eval",evalutation_loss,epoch)

            if evalutation_loss < evalutation_best_performance:
                evalutation_best_performance = evalutation_loss
                self.save_model(epoch, len(self.train_loader))
            else:
                print("The model did not improve and has started to overfit, the best performance was: {}".format(evalutation_best_performance))
                break

    def evaluate(self):
        print("Evaluating...")
        self.model.eval()

        running_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader,0):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.criterion(output, y)

                running_loss+= loss.item()
        
        return running_loss / len(self.test_loader)

    def predict(self,x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self.model(x)
            return output

