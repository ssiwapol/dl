import os
import os.path as osp
import csv

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

from letter import LETTER_LIST


class LibriTrain(Dataset):

    def __init__(self, dir):
        # TODO
        # Join path for features and label
        self.X_dir = osp.join(dir, 'mfcc')
        self.Y_dir = osp.join(dir, 'transcript')

        # Get all files in the directory
        self.X_files = [osp.join(self.X_dir, i) for i in os.listdir(self.X_dir)]
        self.Y_files = [osp.join(self.Y_dir, i) for i in os.listdir(self.Y_dir)]

        # Set LETTER_LIST for mapping index
        self.LETTER_LIST = LETTER_LIST
        self.output_size = len(LETTER_LIST)

        # Verify number of files
        assert(len(self.X_files) == len(self.Y_files))
        

    def __len__(self):
        # TODO
        return len(self.X_files)

    def __getitem__(self, ind):
        # TODO

        # Load data
        X = np.load(self.X_files[ind])
        Y = np.load(self.Y_files[ind])

        # Map LETTER_LIST for mapping index
        Yy = np.array([self.LETTER_LIST.index(i) for i in Y])
        
        # Return x and y
        return torch.tensor(X), torch.tensor(Yy, dtype=torch.long)
    
    def collate_fn(batch):
        # TODO

        # Concat batch input
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        # Pad the sequence to the max_length
        batch_x_pad = pad_sequence(batch_x, batch_first=True)
        # Get original length before padding
        lengths_x = [i.shape[0] for i in batch_x]

        # Pad the sequence to the max_length
        batch_y_pad = pad_sequence(batch_y, batch_first=True)
         # Get original length before padding
        lengths_y = [i.shape[0] for i in batch_y]

        # Return batch x, y, length_x, length_y
        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


class LibriTest(Dataset):

    def __init__(self, dir, test_order='test_order.csv'):
        # TODO

        # Read the order from the csv file
        with open(osp.join(dir, test_order), 'r') as f:
            data = list(csv.reader(f))
        test_order_list = [i[0] for i in data[1:]]

        # Set output size
        self.output_size = len(LETTER_LIST)

        # Get all files in the directory ordered by csv file
        self.X_files = [osp.join(dir, 'mfcc', i) for i in test_order_list]
    
    def __len__(self):
        # TODO
        return len(self.X_files)
    
    def __getitem__(self, ind):
        # TODO

        # Load data
        X = np.load(self.X_files[ind])

        # Return x
        return torch.tensor(X)
    
    def collate_fn(batch):
        # TODO

        # Concat batch input
        batch_x = [x for x in batch]
        # Pad the sequence to the max_length
        batch_x_pad = pad_sequence(batch_x, batch_first=True)
        # Get original length before padding
        lengths_x = [i.shape[0] for i in batch_x]

        # Return batch x, length_x
        return batch_x_pad, torch.tensor(lengths_x)


class LibriTrainData():
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
    
    def load(self, train=True):

        # Set dataset from LibriTrain
        self.dataset = LibriTrain(self.dir)

        # Create DataLoader
        shuffle = True if train else False
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=shuffle, collate_fn=LibriTrain.collate_fn)
        
        # Find input size and output size
        self.input_size = self.dataset[0][0].shape[1]
        self.output_size = self.dataset.output_size

        return self.data_loader

    @staticmethod
    def find_n(dir):
        dataset = LibriTrain(dir)
        input_size = dataset[0][0].shape[1]
        output_size = dataset.output_size
        return input_size, output_size


class LibriTestData():
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
    
    def load(self):

        # Set dataset from LibriTest
        self.dataset = LibriTest(self.dir)

        # Create DataLoader
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, 
            shuffle=False, collate_fn=LibriTest.collate_fn
            )
        
        # Find input size and output size
        self.input_size = self.dataset[0].shape[1]
        self.output_size = self.dataset.output_size

        return self.data_loader
