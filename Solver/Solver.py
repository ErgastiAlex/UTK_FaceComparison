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
            #TODO add weight decay
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)


        self.writer = writer



    def save_model(self, epoch):
        # if you want to save the model
        if not os.path.exists(self.args.checkpoint_path):
            os.mkdir(self.args.checkpoint_path)
        
        path = os.path.join(self.args.checkpoint_path, 
                                    self.model.__class__.__name__ +"_"+str(epoch)+".pth")

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

                n_iter=epoch * len(self.train_loader) + i
                self.writer.add_scalar('Loss/train', loss.item(), n_iter)

                #TODO maybe add some other metrics to the writer
                if i % self.args.print_every == 0:
                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss.item()))

                self.writer.flush()

            self.save_model(epoch)

        #TODO check if this is the right place to close the writer
        self.writer.close()