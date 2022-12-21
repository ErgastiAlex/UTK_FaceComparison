import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
#TODO Fix this
from Dataset.UTKDataset import UTKDataset
from Solver.Solver import Solver
from torch.utils.data import DataLoader
from Model.SiameseResNet import SiameseResNet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        default="run_1", help='name of current run')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')

    parser.add_argument('--workers', type=int, default=1, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')

    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    parser.add_argument('--criterion', type=str, default='BCELoss', choices=['BCELoss', 'MSELoss'], help = 'criterion used for training')

    # parser.add_argument('--use_norm', action='store_true', help='use normalization layers in model')
    # parser.add_argument('--feat', type=int, default=16, help='number of features in model')


    parser.add_argument('--dataset_path', type=str, default='./UTKFace', help='path were to get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./models', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')
    parser.add_argument('--seed', type=int, default=42, help='seed for the random number generator')
    parser.add_argument('--year_diff', type=int, default=1, help='minimum age difference between the two images')
    parser.add_argument('--train_size', type=int, default=1000, help='number of images to use to generate the training dataset. If it is greater than the number of images in the directory, it will be clamped to the number of images in the directory')
    parser.add_argument('--test_size', type=int, default=100, help='number of images to use to generate the test dataset.')

    return parser.parse_args()


def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    # Load the dataset
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset=UTKDataset(root_dir=args.dataset_path, transform=transform, seed=args.seed, year_diff=args.year_diff, data_size=args.train_size)
    test_dataset=UTKDataset(root_dir=args.dataset_path, transform=transform, seed=args.seed, year_diff=args.year_diff, data_size=args.test_size,exclude_images=train_dataset.get_images())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ")
    print(device)

    solver= Solver(train_loader,test_loader,device,SiameseResNet(),writer,args)

    solver.train()

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
