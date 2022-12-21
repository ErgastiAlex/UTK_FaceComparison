import glob
import argparse
import numpy as np
import re
import shutil

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='D:\\Alex\\Universita\\Parma - Scienze Informatiche\\2INF\\Deep Learning and Generative Models\\Progetto\\UTKFace\\dataset', help='path to the dataset')
    parser.add_argument('--trainset_path', type=str, default='D:\\Alex\\Universita\\Parma - Scienze Informatiche\\2INF\\Deep Learning and Generative Models\\Progetto\\UTKFace\\train', help='path to the trainset')
    parser.add_argument('--testset_path', type=str, default='D:\\Alex\\Universita\\Parma - Scienze Informatiche\\2INF\\Deep Learning and Generative Models\\Progetto\\UTKFace\\test', help='path to the testset')

    parser.add_argument('--split_percentage', type=float, default=0.8, help='percentage of the dataset to use for the training set')

    return parser.parse_args()


def main():
    args = get_args()

    train_images=np.empty(0)
    test_images=np.empty(0)

    images=np.array(glob.glob(args.dataset_path + '\\*.jpg'))

    ages=np.vectorize(lambda x: re.search("(\d+)_\d_\d_\d+\.jpg.*",x))(images)
    images_data=np.c_[images,ages]

    images_data=images_data[images_data[:,1]!=None]
    images_data[:,1]=np.vectorize(lambda x: int(x.group(1)))(images_data[:,1])

    age_count=np.unique(images_data[:,1], return_counts=True)

    # select the same ratio of images for each age inside the train and test set
    for i in range(len(age_count[0])):
        age=age_count[0][i]
        count=age_count[1][i]

        train_count=int(count*args.split_percentage)


        train_images=np.append(train_images, images_data[images_data[:,1]==age][:train_count,0])
        test_images=np.append(test_images, images_data[images_data[:,1]==age][train_count:,0])

    # save the train and test images
    for image in train_images:
        shutil.copy(image, args.trainset_path+'\\'+image.split('\\')[-1])

    for image in test_images:
        shutil.copy(image, args.testset_path+'\\'+image.split('\\')[-1])

    
    

if __name__ == '__main__':
    args = get_args()

    main()