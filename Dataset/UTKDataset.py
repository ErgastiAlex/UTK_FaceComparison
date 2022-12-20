import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
from itertools import permutations 
import random
from PIL import Image
import re


class UTKDataset(Dataset):
    def __init__(self,  root_dir:str, transform=None, seed:int=42, year_diff:int =1, data_size:int = 1000, exclude_images=[]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Seed for the random number generator
            year_diff (int): Minimum age difference between the two images
            data_size (int): Number of images to use to generate the dataset. If it is greater than the number of images in the directory, it will be clamped to the number of images in the directory
        """

        random.seed(seed)

        self.root_dir = root_dir
        self.transform = transform
        self.year_diff = year_diff

        self.__get_all_images_in_dir(root_dir)
        self.exclude_images=exclude_images
        
        if data_size<=0:
            raise Exception("Data size must be greater than 0")
        
        self.data_size=data_size
        self.__clamp_data_size()
        self.__select_images_randomly()

        if(len(self.files)==0):
            raise Exception("No images found in the directory")
        
        self.__create_dataset()


    def __get_all_images_in_dir(self, root_dir):
        """
        Get all the images in the directory
        """
        if root_dir[-1]=="/":
            self.files = glob.glob(root_dir + '*.jpg')
        else:
            self.files = glob.glob(root_dir + '/*.jpg')
    

    def __clamp_data_size(self):
        """
        Clamp the data size to the number of images in the directory if it is greater
        """
        if self.data_size>len(self.files):
            self.data_size=len(self.files)

 
        

    def __select_images_randomly(self):
        """
        Select @data_size images randomly from the directory
        """
        self.files=list(filter(lambda x: x not in self.exclude_images, self.files))
        self.files=random.sample(self.files, self.data_size)


    def __create_dataset(self):
        self.images=map(lambda x: (x,re.search("(\d+)_\d_\d_\d+\.jpg.*",x)), self.files)

        #remove all non-matching images, just in case!
        self.images=filter(lambda x: x[1]!=None, self.images)

        #This must be a list, otherwise DataLoader will not work
        self.images=list(map(lambda x: (x[0],x[1].group(1)), self.images))

        # Create all possible combinations of images
        image_combinations = permutations(self.images, 2)
        
        # Filter out images with age difference less than year_diff
        image_combinations=filter(lambda x: abs(int(x[0][1]) - int(x[1][1])) >= self.year_diff, image_combinations)


        # Create a list of tuples ((image1, image2), label)
        self.data= list(map(lambda x: ((x[0][0], x[1][0]), 
                                    1 if int(x[0][1]) > int(x[1][1]) else 0), image_combinations))
                      

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        img0 = UTKDataset.__load_image(self.data[idx][0][0])
        img1 = UTKDataset.__load_image(self.data[idx][0][1])

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        else:
            toTensor=transforms.ToTensor()
            img0=toTensor(img0)
            img1=toTensor(img1)

        return torch.stack((img0,img1),0), self.data[idx][1]
        

    def __load_image(img_name):
        with open(img_name, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def get_images(self):
        return self.images


def main():
    import matplotlib.pyplot as plt
    import os

    dataloader=UTKDataset(root_dir=os.getcwd()+"\\UTKFace", year_diff=1)
    

    # Visualize the data
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataloader), size=(1,)).item()
        images, label = dataloader[sample_idx]
        img0=images[0]
        img1=images[1]

        figure.add_subplot(rows, cols, i)
        plt.title(f"Left is older?: {label}")

        plt.axis("off")

        plt.imshow(transforms.ToPILImage()(torch.cat((img0,img1),dim=2)))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()