import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
from PIL import Image
import re
import numpy as np

class UTKAgeDataset(Dataset):
    def __init__(self,  root_dir:str, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Seed for the random number generator
            year_diff (int): Minimum age difference between the two images
            data_size (int): Size of the dataset, if None or negative the dataset will be the maximum possible, if unique_images is True, the dataset required size could not be reached
            duplicate_probability (float): [0..1] Probability of duplication a combination of images by switching the images order, the duplication will return a greater dataset size by a factor of 1+duplicate_probability
            unique_images (bool): If True, an image will be used only in one combination
            return_image_age (bool): If True, the __getitem__ function will return the age of the images
        """

        self.transform = transform

        self.files=self.__get_all_images_in_dir(root_dir)
        if(len(self.files)==0):
            raise Exception("No images found in the directory")

        # Data that will be returned by the __getitem__ function
        self.images_data= np.empty(0) 
        
        self.__create_dataset()

    def __get_all_images_in_dir(self, root_dir:str)->np.array:
        """
        Get all the images in the directory
        """
        if root_dir[-1]=="/":
            return np.array(glob.glob(root_dir + '*.jpg'))
        else:
            return np.array(glob.glob(root_dir + '/*.jpg'))


    def __create_dataset(self):
        #Get ages for all images
        ages=self.__get_ages()

        # Combina images name and age
        self.__get_images_data(ages)


    def __get_ages(self):
        """Return the age of each image"""
        return np.vectorize(lambda x: re.search("(\d+)_\d_\d_\d+\.jpg.*",x))(self.files)

    def __get_images_data(self,ages):
        """Create images_data, images name and age cleaned from None values and converted to int"""

        # Combina images name and age
        self.images_data=np.c_[self.files,ages]

        #remove all images that do not have an age, just to be safe
        self.images_data=self.images_data[self.images_data[:,1]!=None]
        #Convert re.match to int
        self.images_data[:,1]=np.vectorize(lambda x: int(x.group(1)))(self.images_data[:,1])


    def __len__(self):
        return len(self.images_data)

    
    def __getitem__(self, idx):
        img = UTKAgeDataset.__load_image(self.images_data[idx][0])
        age=self.images_data[idx][1]

        if self.transform:
            img = self.transform(img)
        else:
            toTensor=transforms.ToTensor()
            img=toTensor(img)

        
        
        return img, torch.tensor(age)
        

    def __load_image(img_name):
        with open(img_name, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def get_ages(self):
        ages=self.images_data[:,1].reshape(-1)
        return ages
    


def main():
    import matplotlib.pyplot as plt
    import os

    dataloader=UTKAgeDataset(root_dir=os.getcwd()+"/UTKFace/train")
    

    # Visualize the data
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataloader), size=(1,)).item()
        image,age= dataloader[sample_idx]


        figure.add_subplot(rows, cols, i)
        plt.title(f"Age: {age.item()}")

        plt.axis("off")

        plt.imshow(transforms.ToPILImage()(image))

    plt.tight_layout()
    plt.show()

    print(f"Dataloader size={len(dataloader)}")

    test_dataloader=UTKAgeDataset(root_dir=os.getcwd()+"/UTKFace/test")

    print(f"Test dataloader size={len(test_dataloader)}")

    train_ages=dataloader.get_ages()

    test_ages=test_dataloader.get_ages()

    
    plot_hist_comparison(plt, train_ages, test_ages, "age distribution")


def plot_hist_comparison(plt, train, test, suffix):
    figure = plt.figure(figsize=(8, 8))

    figure.add_subplot(1, 2, 1)
    plt.hist(train, bins=range(min(train), max(train) + 1, 1))
    plt.title("Train "+suffix)

    figure.add_subplot(1, 2, 2)
    plt.hist(test, bins=range(min(test), max(test) + 1, 1))
    plt.title("Test "+suffix)
    plt.show()


if __name__ == "__main__":
    main()