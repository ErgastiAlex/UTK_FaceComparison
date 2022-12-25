import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

class AutoSolver():
    """Solver for find the best hyperparameters for the model, automatically."""

    def __init__(self, training_set, test_set,val_set , model_class, writer, args):
        
        self.args = args
        self.model_path=os.path.join(args.checkpoint_path,args.run_name)

        self.model_class = model_class
        self.trainset = training_set
        self.testset = test_set
        self.valset= val_set

        self.config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([4,8,16, 32, 64,128]),
            "hidden_layers": tune.choice([[],[64, 32, 16], [128, 64, 32], [256, 128, 64], [512, 256, 128]]),
            "use_dropout": tune.choice([True, False]),
            "dropout_prob": tune.uniform(0.1, 0.7),
            "weight_decay": tune.loguniform(1e-6, 1e-2),
            "resnet_type": tune.choice([torchvision.models.resnet18, torchvision.models.resnet18, torchvision.models.resnet18])
        }

    def start_search(self,num_samples=10, max_num_epochs=10, gpus_per_trial=2):   
        """Start the search for the best hyperparameters for the model."""

        #At each trial, Tune will now randomly sample a combination of parameters from these search spaces. 
        # It will then train a number of models in parallel and find the best performing one among these. 
        # We also use the ASHAScheduler which will terminate bad performing trials early.

        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.__train),
                resources={"cpu": 2, "gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=self.config,
        )

        results = tuner.fit()

        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config),flush=True)
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]),flush=True)
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]),flush=True)

        self.__test_best_model(best_result)
    

    def __train(self,config):
        net = self.model_class(hidden_layers=config["hidden_layers"], use_dropout=config["use_dropout"], dropout_p=config["dropout_prob"], resnet_type=config["resnet_type"])

        device = "cpu"

        # If there are GPUs available, use them.
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net) # the module is copied to each GPU and each one handles a portion of the batch

        net.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])

        # To restore a checkpoint, use `session.get_checkpoint()`.
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
               model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        data_dir = os.path.abspath("./data")


        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=8)

        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=8)

        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1

                if i % self.args.print_every == self.args.print_every-1: 
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)

                    predicted=torch.gt(outputs,0.5).int()
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and can be accessed through `session.get_checkpoint()`
            # API in future iterations.
            os.makedirs(self.model_path, exist_ok=True)
            torch.save(
                (net.state_dict(), optimizer.state_dict()), self.model_path+"/checkpoint.pt")

            checkpoint = Checkpoint.from_directory(self.model_path)
            session.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)

        print("Finished Training")

    def __test_best_model(self,best_result):
        # Load the best model and test it on the test set.
        config=best_result.config
        best_trained_model = self.model_class(hidden_layers=config["hidden_layers"], use_dropout=config["use_dropout"], dropout_p=config["dropout_prob"], resnet_type=config["resnet_type"])

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        best_trained_model.to(device)

        checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

        # Load the best model checkpoint.
        model_state, optimizer_state = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state)

        

        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = best_trained_model(images)
                predicted=torch.gt(outputs,0.5).int()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print("Best trial test set accuracy: {}".format(correct / total))