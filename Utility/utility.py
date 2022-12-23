import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

def display_dataset_info(writer, dataset, nrow, n_images,prefix=""):
    """
    Display some images from the dataset and distribution of the ages in the dataset
    """
    writer.add_text(f"{prefix} dataset size", str(len(dataset)))


    show_dataset_image_example(dataset, nrow, n_images, writer)
    show_dataset_statistics(dataset, writer)


def show_dataset_statistics(dataset, writer):
    """
    Display the distribution of the ages and the ages difference
    """

    # Get the ages and the ages difference and display them
    ages=np.array(dataset.get_ages())
    ages_diff=np.array(dataset.get_ages_diff())

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



def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)