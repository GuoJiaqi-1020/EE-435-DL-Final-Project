import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from math import ceil

def get_train_val_test_dataloader(args):
    dataset = FaceMaskDataset(args.data_dir)

    train_val_size = ceil(args.train_val_proportion * len(dataset))
    train_val_ds, test_ds = random_split(dataset, [train_val_size, len(dataset) - train_val_size], generator=torch.Generator().manual_seed(args.seed))
    train_size = ceil(args.train_proportion * len(dataset))
    train_ds, val_ds = random_split(train_val_ds, [train_size, train_val_size - train_size], generator=torch.Generator().manual_seed(args.seed))

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_ds ,batch_size=args.batch_size)
    return train_dataloader, val_dataloader, test_dataloader

class FaceMaskDataset(Dataset):
    def __init__(self, data_dir, transform=ToTensor(), target_transform=None):
        '''
        data_dir: data direction name used for training.
        classes = [Correct, InCorrect, NoMask]
        '''
        self.transform = transform
        self.target_transform = target_transform

        data_correct = np.load(data_dir+'Correct.npy')
        data_incorrect = np.load(data_dir+'InCorrect.npy')
        data_nomask = np.load(data_dir+'NoMask.npy')

        self.x = np.concatenate((data_correct, data_incorrect, data_nomask), 0)
        self.y = torch.cat((torch.zeros(len(data_correct)), torch.ones(len(data_incorrect)), torch.ones(len(data_nomask))*2), 0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_ = self.x[idx]
        y_ = self.y[idx]

        if self.transform:
            x_ = self.transform(x_)
        if self.target_transform:
            y_ = self.target_transform(y_)
        return x_.type(torch.float), y_.type(torch.LongTensor)