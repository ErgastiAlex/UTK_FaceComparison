import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
from PIL import Image
import re
import numpy as np

class UTKDataset(Dataset):
    def __init__(self,  root_dir:str, transform=None, seed:int=42, year_diff:int =1,  data_size:int = None, duplicate_probability=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Seed for the random number generator
            year_diff (int): Minimum age difference between the two images
            data_size (int): Size of the dataset, if None or negative the dataset will be the maximum possible
            duplicate_probability (float): Probability of duplication a combination of images by switching the images order, the duplication will return a greater dataset size by a factor of 1+duplicate_probability
        """
        np.random.seed(seed)

        self.data_size=data_size

        self.transform = transform


        self.files=self.__get_all_images_in_dir(root_dir)
        if(len(self.files)==0):
            raise Exception("No images found in the directory")

        # Data that will be returned by the __getitem__ function
        self.data= np.empty(0) 

        # Np array with images name and age
        self.images_data=np.empty(0) 

        # Np array with the age difference between combination of images in the dataset
        self.age_diff=np.empty(0)  
        
        self.__create_dataset(year_diff,duplicate_probability)


    def __get_all_images_in_dir(self, root_dir:str)->np.array:
        """
        Get all the images in the directory
        """
        if root_dir[-1]=="/":
            return np.array(glob.glob(root_dir + '*.jpg'))
        else:
            return np.array(glob.glob(root_dir + '/*.jpg'))


    def __create_dataset(self,year_diff:int,duplicate_probability:float):
        #Get ages for all images
        ages=np.vectorize(lambda x: re.search("(\d+)_\d_\d_\d+\.jpg.*",x))(self.files)

        # Combina images name and age
        self.images_data=np.c_[self.files,ages]

        #remove all images that do not have an age, just to be safe
        self.images_data=self.images_data[self.images_data[:,1]!=None]

        #Convert re.match to int
        self.images_data[:,1]=np.vectorize(lambda x: int(x.group(1)))(self.images_data[:,1])

        self.__get_images_combinations(year_diff,duplicate_probability)
        
                      

    def __get_images_combinations(self,year_diff,duplicate_probability):
        """
        Extract @data_size number of combinations of images
        """
        ages=self.images_data[:,1].astype(int)
        
        #Get combinations of images where the age difference is greater than year_diff
        #ages[:,None] has shape (ages.shape[0],1)
        #The comparison between a matrix and a vector is done by brodcasting.
        #ages[:,None] is broadcasted to (ages.shape[0],ages.shape[0]) by duplicating the column
        #ages + year_diff is broadcasted from (ages.shape[0],) to (ages.shape[0],ages.shape[0]) by duplicating the vector in each row

        r,c=np.nonzero(ages[:,None] + year_diff <= ages)

        #r selects the rows of the matrix where the condition is true, c selects the columns of the matrix where the condition is true
        #so r and c will contain the indexes of each combination of images where the age difference is greater than year_diff
        combinations_indexes=np.c_[r,c]

        if self.data_size is not None and self.data_size>0 and combinations_indexes.shape[0]<self.data_size:
            self.data_size=combinations_indexes.shape[0]
            print("WARNING: The dataset size is greater than the maximum possible size, the dataset size will be the maximum possible size")
        elif self.data_size is None or self.data_size<0:
            self.data_size=combinations_indexes.shape[0]
            print("WARNING: The dataset size is not specified, the dataset size will be the maximum possible size")
        
        combinations_selected=np.random.choice(combinations_indexes.shape[0],self.data_size,replace=False)

        #Get the indexes of the images in the dataset
        images1_index=combinations_indexes[combinations_selected,0].astype(int)
        images2_index=combinations_indexes[combinations_selected,1].astype(int)

        images1=self.images_data[images1_index,0]
        images2=self.images_data[images2_index,0]
        left_is_older=(ages[images1_index]>ages[images2_index]).astype(int)

        self.__switch_images(images1,images2,left_is_older)

        self.age_diff=abs(ages[images1_index]-ages[images2_index])



        images1,images2,left_is_older=self.__duplicate_images(images1,images2,left_is_older,duplicate_probability,ages)

        # Create the data that will be returned by the __getitem__ function
        self.data=np.c_[images1,images2,left_is_older]
    
    
    def __switch_images(self,images1,images2,left_is_older):
        """Switch half of the images order in the dataset"""
        switch_images_index=np.random.choice(self.data_size,self.data_size//2,replace=False)
        images1[switch_images_index],images2[switch_images_index]=images2[switch_images_index],images1[switch_images_index]

        #Set the label to 0 for the images that were switched
        left_is_older[switch_images_index]=1-left_is_older[switch_images_index]

    def __duplicate_images(self,images1,images2,left_is_older,duplicate_probability,ages):
        """Duplicate @duplicate_probability of the images in the dataset"""
        duplicate_datasize=int(self.data_size*duplicate_probability)

        if duplicate_datasize==0:
            return images1,images2,left_is_older

        duplicate_images_index=np.random.choice(self.data_size,duplicate_datasize,replace=False)
        
        images1=np.append(images1,images2[duplicate_images_index])
        images2=np.append(images2,images1[duplicate_images_index])
        left_is_older=np.append(left_is_older,1-left_is_older[duplicate_images_index])

        self.age_diff=np.append(self.age_diff,abs(ages[duplicate_images_index]-ages[duplicate_images_index]))

        return images1,images2,left_is_older

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        img0 = UTKDataset.__load_image(self.data[idx][0])
        img1 = UTKDataset.__load_image(self.data[idx][1])

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        else:
            toTensor=transforms.ToTensor()
            img0=toTensor(img0)
            img1=toTensor(img1)

        # get the label
        labels=torch.tensor(self.data[idx][2])
        labels=labels.unsqueeze(-1).to(torch.float32) # [x] -> [x,1] and convert to float32

        images=torch.stack((img0,img1),dim=0)
        
        return images, labels
        

    def __load_image(img_name):
        with open(img_name, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img

    def get_ages(self):
        ages=list(map(lambda x: int(x[1]), self.images_data))
        return ages
    
    def get_ages_diff(self):
        return self.age_diff


def main():
    import matplotlib.pyplot as plt
    import os

    dataloader=UTKDataset(root_dir=os.getcwd()+"\\UTKFace\\train", year_diff=1,data_size=100000)
    

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

    test_dataloader=UTKDataset(root_dir=os.getcwd()+"\\UTKFace\\test", year_diff=1,data_size=1000)

    print(f"Test dataloader size={len(test_dataloader)}")


    train_ages=dataloader.get_ages()
    train_age_diffs=dataloader.get_ages_diff()

    test_ages=test_dataloader.get_ages()
    test_age_diffs=test_dataloader.get_ages_diff()
    
    plot_hist_comparison(plt, train_ages, test_ages, "age distribution")
    plot_hist_comparison(plt, train_age_diffs, test_age_diffs, "age difference distribution")

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