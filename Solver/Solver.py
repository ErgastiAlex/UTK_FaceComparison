import torch
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
import copy
import numpy as np

class Solver():
    """Solver for training and testing a model with parameters defined by the user in the config file."""
    def __init__(self, train_loader, test_loader, device, model, writer, args):
        self.args = args

        self.model_path=os.path.join(self.args.checkpoint_path,self.args.run_name)

        self.epochs = args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Patience for early stopping, ask the user to input it
        self.patience=args.patience
        self.model = model
        
        self.device = device 

        # Don't load the model in DataParallel if we are testing here, it will be done inside load_model
        if torch.cuda.device_count() > 1 and args.test==False:
                self.model = torch.nn.DataParallel(self.model) # the module is copied to each GPU and each one handles a portion of the batch
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

        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        if not os.path.exists(self.model_path):
            raise "No checkpoint found!"

        if self.model_path[-1] == '/':
            self.model_path = self.model_path[:-1]
        
        saved_models= glob.glob(self.model_path + '/*.pth')

        if len(saved_models) == 0:
            raise "No model found!"

        # get the last model saved in the folder
        last_model= max(saved_models, key=os.path.getctime)

        epoch, iteration= 0,0
        try:
            epoch, iteration = last_model.split("_")[-2:]
            iteration=iteration.split(".")[0]
            epoch=int(epoch)
            iteration=int(iteration)
        except:
            #The model was saved without any epoch and iteration information
            pass
        
        self.model.load_state_dict(torch.load(last_model))

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            

        print("Model loaded!")

        return epoch,iteration


    def train(self):
        print("Training...")

        evalutation_best_loss = 100000
        evalutation_best_accuracy= -1
        best_model=None

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


                    print("Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}".format(epoch, i, running_loss / self.args.print_every, accuracy / self.args.print_every), flush=True)
                    running_loss = 0.0
                    accuracy = 0.0
                    
                    self.save_model(epoch, i)

                self.writer.flush()


            self.save_model(epoch, len(self.train_loader))
            # Early stopping 
            evalutation_loss, evaluation_accuracy, evaluation_auc_score= self.evaluate()


            self.writer.add_scalar("Loss/eval",evalutation_loss,epoch)
            self.writer.add_scalar("Accuracy/eval",evaluation_accuracy,epoch)
            self.writer.add_scalar("AUC/eval",evaluation_auc_score,epoch)

            # For early stopping we use the loss on the validation set not the accuracy
            if evalutation_loss < evalutation_best_loss:
                print("Better model found, old loss: {}, new loss: {}".format(evalutation_best_loss, evalutation_loss))

                evalutation_best_loss = evalutation_loss
                evalutation_best_accuracy = evaluation_accuracy
                best_model = copy.deepcopy(self.model)

                patience_counter=0
                # Save only the best model

            else:
                patience_counter+=1
                print(f"Keep searching for the best model, patience counter: {patience_counter}")
                # If we have reached the patience we stop the training
                if patience_counter == self.patience:
                    print("The model did not improve and has started to overfit, the best performances are with loss of: {}  and accuracy of: {}".format(evalutation_best_loss, evalutation_best_accuracy))
                    break
        
        self.model=best_model
        #Save the best model as the last model
        self.save_model(self.epochs+1, 0) 


    def evaluate(self, write_to_tensorboard=False):
        print("Testing...")
        self.model.eval()

        running_loss = 0.0
        accuracy = 0.0

        y_pred=np.array([])
        y_true=np.array([])
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader,0):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.criterion(output, y)

                running_loss+= loss.item()
                accuracy += (torch.gt(output,0.5).int() == y).sum().item() / y.shape[0]

                # Append the predictions and the true labels to compute the ROC curve
                y_pred=np.append(y_pred,output.cpu().numpy())
                y_true=np.append(y_true,y.cpu().numpy())

        auc_score=0
        try:
            auc_score=roc_auc_score(y_true,y_pred)
        except:
            #not always possible to calculate the AUC
            pass
        
        print("Test loss: {}, Test accuracy: {} Test AUC: {}".format(running_loss / len(self.test_loader), accuracy / len(self.test_loader), auc_score))

        if write_to_tensorboard:
            self.writer.add_text("Test loss", str(running_loss / len(self.test_loader)))
            self.writer.add_text("Test accuracy", str(accuracy / len(self.test_loader)))
            self.writer.add_text("Test AUC", str(auc_score))

            fpr, tpr, thresholds=roc_curve(y_true,y_pred)

            figure=plt.figure(figsize=(10, 10))
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_score)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            self.writer.add_figure('ROC', figure)
            self.writer.flush()

        return running_loss / len(self.test_loader), accuracy / len(self.test_loader), auc_score

            
