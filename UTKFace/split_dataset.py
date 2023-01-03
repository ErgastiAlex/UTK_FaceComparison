import glob
import argparse
import numpy as np
import re
import shutil
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--dataset_path', type=str, default='D:/Alex/Universita/Parma - Scienze Informatiche/2INF/Deep Learning and Generative Models/Progetto/UTKFace/dataset', help='path to the dataset')
    parser.add_argument('--trainset_path', type=str, default='D:/Alex/Universita/Parma - Scienze Informatiche/2INF/Deep Learning and Generative Models/Progetto/UTKFace/train', help='path to the trainset')
    parser.add_argument('--valset_path', type=str, default='D:/Alex/Universita/Parma - Scienze Informatiche/2INF/Deep Learning and Generative Models/Progetto/UTKFace/val', help='path to the validation set')
    parser.add_argument('--testset_path', type=str, default='D:/Alex/Universita/Parma - Scienze Informatiche/2INF/Deep Learning and Generative Models/Progetto/UTKFace/test', help='path to the testset')

    parser.add_argument('--train_size', type=int, default=70, help='percentage of the dataset to use for the training set')
    parser.add_argument('--val_size', type=int, default=20, help='percentage of the dataset to use for the validation set')
    parser.add_argument('--test_size', type=int, default=10, help='percentage of the dataset to use for the test set')
    
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    assert args.train_size + args.val_size + args.test_size == 100, "The sum of the train, val and test size must be 1"

    np.random.seed(args.seed)

    train_images=np.empty(0)
    val_images=np.empty(0)
    test_images=np.empty(0)

    images_path=os.path.join(args.dataset_path, '*.jpg')
    images=np.array(glob.glob(images_path))
    np.random.shuffle(images)

    ages=np.vectorize(lambda x: re.search("(\d+)_\d_\d_\d+\.jpg.*",x))(images)
    images_data=np.c_[images,ages]

    images_data=images_data[images_data[:,1]!=None]
    images_data[:,1]=np.vectorize(lambda x: int(x.group(1)))(images_data[:,1])

    age_count=np.unique(images_data[:,1], return_counts=True)

    # select the same ratio of images for each age inside the train and test set
    for i in range(len(age_count[0])):
        age=age_count[0][i]
        count=age_count[1][i]

        train_count=int(count*args.train_size/100)
        val_count=int(count*args.val_size/100)

        


        train_images=np.append(train_images, images_data[images_data[:,1]==age][:train_count,0])
        val_images=np.append(val_images, images_data[images_data[:,1]==age][train_count:train_count+val_count,0])
        test_images=np.append(test_images, images_data[images_data[:,1]==age][train_count+val_count:,0])

    print("Deleting old images...")
    # remove the old train, val and test images
    if os.path.exists(args.trainset_path):
        shutil.rmtree(args.trainset_path)

    if os.path.exists(args.valset_path):
        shutil.rmtree(args.valset_path)

    if os.path.exists(args.testset_path):
        shutil.rmtree(args.testset_path)
    
    os.mkdir(args.trainset_path)
    os.mkdir(args.valset_path)
    os.mkdir(args.testset_path)

    print("Saving images...")
    # save the train and test images
    for image in train_images:
        image_name=os.path.split(image)[-1]
        image_path=os.path.join(args.trainset_path, image_name)
        shutil.copy(image, image_path)

    for image in val_images:
        image_name=os.path.split(image)[-1]
        image_path=os.path.join(args.valset_path, image_name)
        shutil.copy(image, image_path)

    for image in test_images:
        image_name=os.path.split(image)[-1]
        image_path=os.path.join(args.testset_path, image_name)
        shutil.copy(image, image_path)

    
    

if __name__ == '__main__':
    args = get_args()

    main()