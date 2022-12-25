import torch
import torchvision.transforms as transforms
from torchvision import models
import argparse, configparser


from torch.utils.tensorboard import SummaryWriter

from Dataset.UTKAgeDataset import UTKAgeDataset
from Solver.Solver import Solver

from torch.utils.data import DataLoader
from Model.ResNetAgeClassifier import ResNetAgeClassifier
import Utility.utility as utility
from localconfig import config

import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=False, help='path to config file')

    parser.add_argument('--run_name', type=str,
                        default="exp1_age_classifier", help='name of current run')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='number of elements in batch size')

    parser.add_argument('--workers', type=int, default=1, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=100, help='print losses every N iteration')


    # Model 
    parser.add_argument('--resnet_type', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='resnet type used for the model')


    #Model parameters
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for the optimizer')

    parser.add_argument('--disable_norm', action='store_true', help="disable normalization of the images")

    # Path parameters
    parser.add_argument('--training_set_path', type=str, default='./UTKFace/train', help='training set path')
    parser.add_argument('--validation_set_path', type=str, default='./UTKFace/val', help='validation set path')
    parser.add_argument('--test_set_path', type=str, default='./UTKFace/test', help='test set path')
    parser.add_argument('--checkpoint_path', type=str, default='./models', help='path were to save the trained model')


    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    # Dataset parameters
    parser.add_argument('--seed', type=int, default=42, help='seed for the random number generator')
    parser.add_argument('--train_size', type=int, default=100000, help='number of images to use to generate the training dataset. If it is greater than the number of images in the directory, it will be clamped to the number of images in the directory')
    parser.add_argument('--validation_size', type=int, default=5000, help='number of images to use to generate the validation dataset.')

    parser.add_argument('--test', action='store_true', help='test the model on the test set')

    args=parser.parse_args()

    if args.config:
        # config = configparser.ConfigParser()
        config.read(args.config)

        defaults = {}
        defaults.update(dict(config.items("Defaults")))
        parser.set_defaults(**defaults)
        args = parser.parse_args() # Overwrite arguments

    args.criterion="MSELoss"    # Force usage of MSE loss for age regression

    return args


def main(args):

    if args.test==True:
        test_model(args)
    else:
        train_model(args)


def test_model(args):
    writer=SummaryWriter('./runs/test/' + args.run_name)

    transform=get_transform(args.disable_norm)

    # Load the dataset 
    test_dataset=UTKAgeDataset(root_dir=args.test_set_path, transform=transform)

    # Create the dataloaders
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ")
    print(device)


    model=ResNetAgeClassifier()

    solver=Solver(None,test_loader,device,model,writer,args)
    solver.load_model()
    solver.test()


def train_model(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    transform=get_transform(args.disable_norm)

    # Load the dataset 
    train_dataset=UTKAgeDataset(root_dir=args.training_set_path, transform=transform)
    validation_dataset=UTKAgeDataset(root_dir=args.validation_set_path, transform=get_transform)

    
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ")
    print(device)

    resnet_class=get_resnet_class(args)
    model=ResNetAgeClassifier(resnet_type=resnet_class)

    solver=Solver(train_loader,validation_loader,device,model,writer,args)
    solver.train()
    

    writer.close()




def get_transform(disable_norm):
    if disable_norm:
        return transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])


def get_resnet_class(args):
    if args.resnet_type == 'resnet18':
        return models.resnet18
    elif args.resnet_type == 'resnet34':
        return models.resnet34
    elif args.resnet_type == 'resnet50':
        return models.resnet50
    elif args.resnet_type == 'resnet101':
        return models.resnet101
    elif args.resnet_type == 'resnet152':
        return models.resnet152
    else:
        raise Exception("Invalid resnet model")



if __name__ == "__main__":
    args = get_args()
    print(args,flush=True)
    main(args)
