import torch
import torchvision
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Model.SiameseResNet import SiameseResNet
from Model.ResNetClassifier import ResNetClassifier
from Model.SiameseResNetAge import SiameseResNetAge
from Model.ResNetAgeClassifier import ResNetAgeClassifier
from torchvision import models

def get_transform(disable_norm):
    """Returns the transform based on the disable_norm flag"""
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


def get_model(args):
    """Returns the model based on the args"""
    resnet_type=get_resnet_class(args.resnet_type)
    if args.model == 'SiameseResNet':
        return SiameseResNet(resnet_type=resnet_type,hidden_layers=args.hidden_layers, use_dropout=args.use_dropout, dropout_p=args.dropout_p)
    elif args.model == 'ResNetClassifier':
        return ResNetClassifier(resnet_type=resnet_type,hidden_layers=args.hidden_layers, use_dropout=args.use_dropout, dropout_p=args.dropout_p)
    elif args.model =='SiameseResNetAge':
        model= SiameseResNetAge(resnet_type=resnet_type,hidden_layers=args.hidden_layers, use_dropout=args.use_dropout, dropout_p=args.dropout_p)
        model.load_model(args.checkpoint_path,args.resnet_type)
        return model
    else:
        raise Exception("Invalid model")


def get_model_class(args):
    """Returns the model class based on the args"""
    if args.model == 'SiameseResNet':
        return SiameseResNet
    elif args.model == 'ResNetClassifier':
        return ResNetClassifier
    elif args.model =='SiameseResNetAge':
        return SiameseResNetAge


def get_resnet_class(resnet_type):
    """Returns the resnet class based on the args"""
    if resnet_type == 'resnet18':
        return models.resnet18
    elif resnet_type == 'resnet50':
        return models.resnet50
        return models.resnet152
    else:
        raise Exception("Invalid resnet model")



def display_dataset_info(writer, dataset, nrow, n_images,prefix=""):
    """
    Display some images from the dataset and distribution of the ages in the dataset
    """
    writer.add_text(f"{prefix} dataset size", str(len(dataset)))


    show_dataset_image_example(dataset, nrow, n_images, writer)

    # Commented out for problemns on HPC
    # show_dataset_statistics(dataset, writer)


def show_dataset_statistics(dataset, writer):
    """
    Display the distribution of the ages and the ages difference
    """

    # Get the ages and the ages difference and display them
    ages=np.array(dataset.get_ages(), dtype=np.int32).flatten()
    ages_diff=np.array(dataset.get_ages_diff(),dtype=np.int32).flatten()

    writer.add_histogram('Ages', ages, bins="auto")
    writer.add_histogram('Ages difference', ages_diff, bins="auto")
    writer.flush()

def show_dataset_image_example(dataset, nrow, n_images, writer):
    """
    Display some images from the dataset
    """
    images=[]
    indexes=torch.randint(len(dataset), size=(n_images,)).tolist()
    for i in indexes:
        images.append(dataset[i][0])

    images=torch.stack(images) # [x, 2, 3, 224, 224]

    images_1,images_2=torch.split(images,1,1) # [x, 2, 3, 224, 224] -> [x, 1, 3, 224, 224] [x, 1, 3, 224, 224]
    images_1=images_1.squeeze(1) # [x, 1, 3, 224, 224] -> [x, 3, 224, 224]
    images_2=images_2.squeeze(1) # [x, 1, 3, 224, 224] -> [x, 3, 224, 224]

    images=torch.cat([images_1,images_2],3) # [x, 3, 224, 224] [x, 3, 224, 224] -> [x, 3, 224, 448]

    # Display the images from the dataset 
    images = torchvision.utils.make_grid(denorm(images[:n_images]), padding=10,nrow=nrow)
    writer.add_image('Images', images)

    writer.flush()

def add_model_info_to_tensorboard(writer, args, model):
    """
    Add args and graph of the model to tensorboard
    """
    
    writer.add_text("Training info", "Training info: \r" + str(args))
    writer.add_graph(model,torch.rand(1,2,3,224,224))

    writer.flush()

def add_image_results_to_tensorboard(writer, model, device, test_dataset, nrows, ncols):
    """
    Add images and the results of the model to tensorboard
    """
    
    figure = plt.figure(figsize=(10, 10))

    for i in range(1,nrows*ncols+1):
        figure.add_subplot(nrows,ncols,i)

        sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
        images,label= test_dataset[sample_idx]

        img0=denorm(images[0])
        img1=denorm(images[1])

        images=images.to(device)
        images=images.unsqueeze(0) # images.unsqueeze(0) -> [1, 2, 3, 224, 224]

        
        prediction=model(images).cpu().view(-1).to(torch.float32)
        label=label.view(-1).to(torch.float32)

        plt.title(f"Pred: {prediction.item():.2f} Real: {label.item():.2f}")
        plt.imshow(transforms.ToPILImage()(torch.cat((img0,img1),dim=2)))

        plt.axis("off")
    
    plt.tight_layout()


    writer.add_figure("Results", figure)
    writer.flush()



def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)