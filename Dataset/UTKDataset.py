import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import random
from PIL import Image
import re


class UTKDataset(Dataset):
    def __init__(self,  root_dir:str, transform=None, seed:int=42, year_diff:int =1, data_size:int = 10000, exclude_images=[]):
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

        self.files=self.__get_all_images_in_dir(root_dir)
        self.exclude_images=exclude_images

        # Remove the images that are in the exclude_images list
        self.files=list(filter(lambda x: x not in self.exclude_images, self.files))
        
        if data_size<=0:
            raise Exception("Data size must be greater than 0")
        
        self.data_size=data_size

        if(len(self.files)==0):
            raise Exception("No images found in the directory")

        self.data=[] # This will contain the data that will be returned by the __getitem__ function
        self.used_images=[] # This will contain the images that have been used to generate the dataset
        self.images_data=[] # This will contain the images and the ages
        
        self.__create_dataset()


    def __get_all_images_in_dir(self, root_dir):
        """
        Get all the images in the directory
        """
        if root_dir[-1]=="/":
            return glob.glob(root_dir + '*.jpg')
        else:
            return glob.glob(root_dir + '/*.jpg')


    def __create_dataset(self):
        #Map the images to a tuple (image_name, age)
        self.images_data=map(lambda x: (x,re.search("(\d+)_\d_\d_\d+\.jpg.*",x)), self.files)

        #remove all images that do not have an age, just to be safe
        self.images_data=filter(lambda x: x[1]!=None, self.images_data)

        #This must be a list, otherwise DataLoader will not work
        self.images_data:tuple(str,int)=list(map(lambda x: (x[0],int(x[1].group(1))), self.images_data))

        self.__get_images_combinations()
        
                      

    def __get_images_combinations(self):
        """
        Extract @data_size number of combinations of images
        """

        # To avoid infinite loops
        tries=0
        max_tries=1000

        # Get all the combinations of images
        while len(self.data) < self.data_size and tries < max_tries:
            img1,age1=random.choice(self.images_data)
            img2,age2=random.choice(self.images_data)

            is_img1_older_than_img2=1 if age1 > age2 else 0

            # Check if the age difference is greater than the year_diff and if the combination has not been added yet
            if abs(age1 - age2) >= self.year_diff and ((img1,img2), is_img1_older_than_img2) not in self.data:

                # Add the combination to the list, ((image1, image2), label)  where label is 1 if image1 is older than image2, 0 otherwise
                self.data.append(((img1,img2),is_img1_older_than_img2))

                self.__add_image_to_used_images(img1)
                self.__add_image_to_used_images(img2)

                # Reset the tries counter
                tries=0
            else:
                tries+=1
            
    def __add_image_to_used_images(self, img):
        if img not in self.used_images:
            self.used_images.append(img)

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

        # get the label
        labels=torch.tensor(self.data[idx][1])
        labels=labels.unsqueeze(-1).to(torch.float32) # [x] -> [x,1] and convert to float32

        images=torch.stack((img0,img1),dim=0)
        
        return images, labels
        

    def __load_image(img_name):
        with open(img_name, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def get_used_images(self):
        return self.used_images

    def get_ages(self):
        ages=list(map(lambda x: int(x[1]), self.images_data))

        return ages
    
    def get_ages_diff(self):
        age_diffs=list(map(lambda x: abs(int(x[0][1]) - int(x[1][1])), self.data))
        return age_diffs


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
        plt.title(f"Left is older?: {label.item()}")

        plt.axis("off")

        plt.imshow(transforms.ToPILImage()(torch.cat((img0,img1),dim=2)))

    plt.tight_layout()
    plt.show()

    print(f"Dataloader size={len(dataloader)}")

    test_dataloader=UTKDataset(root_dir=os.getcwd()+"\\UTKFace", year_diff=1, exclude_images=dataloader.get_used_images())

    print(f"Test dataloader size={len(test_dataloader)}")

    print(f"Intersection test and training set: {list(set(test_dataloader.get_used_images()) & set(dataloader.get_used_images()))}")



if __name__ == "__main__":
    main()