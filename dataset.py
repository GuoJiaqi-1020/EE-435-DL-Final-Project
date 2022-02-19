import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class FaceMaskDataset(Dataset):
    def __init__(self, args, data_dir, transform=None, target_transform=None):
        '''
        data_dir: data direction name used for training.
        classes = [Correct, InCorrect, NoMask]
        '''
        self.data_correct = np.load(data_dir+'Correct.npy')
        self.data_incorrect = np.load(data_dir+'InCorrect.npy')
        self.data_nomask = np.load(data_dir+'NoMask.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return input and label
        # In rnn, if our mission is self regression, the label is the left shift of input. 
        # For example, the sentence is "This is code for our homework". The input should be "This is code for our" , the output should be "is code for our homework".
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return self.x[idx], self.y[idx]