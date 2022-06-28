import os
import os.path as osp

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as ttf


data_aug = [
    ttf.ToTensor(),
    ttf.RandomHorizontalFlip(),
    ttf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ttf.GaussianBlur(kernel_size=(5, 9)),
    ttf.RandomRotation(degrees=20),
    ttf.RandomErasing(p=0.1)
    # Add noise
    ]

data_trans = [
    ttf.ToTensor()
    ]

# Overwrite ImageFolder class to get filepath
class ImageFolderDataSet(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderDataSet, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# Dataset / Dataloader
class ClassificationData():
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
    
    def load(self, train=True, augment=True):
        # Augment images
        if train and augment:
            data_transform = data_aug
        else:
            data_transform = data_trans
        
        # Create dataset
        self.dataset = ImageFolderDataSet(self.dir, transform=ttf.Compose(data_transform))

        # Create data loader (train=shuffle)
        if train:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=True, drop_last=True, num_workers=2)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=False, drop_last=False, num_workers=2)
        
        # Find input size
        x, y, p = iter(self.data_loader).next()
        self.input_size = tuple(x.shape)

        return self.data_loader
    
    @staticmethod
    def find_n(dir, batch_size):
        data_transform = [ttf.ToTensor()]
        dataset = ImageFolderDataSet(dir, transform=ttf.Compose(data_transform))
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=True, num_workers=2)
        return len(dataset.classes), len(data_loader)


# Triple dataset to get anchor, positive and negative
class TripletDataset(torchvision.datasets.VisionDataset):
  def __init__(self, root, transform):
    # For "root", note that you're making this dataset on top of the regular classification dataset.
    self.dataset = ImageFolderDataSet(root=root, transform=transform)
    
    # map class indices to dataset image indices
    self.classes_to_img_indices = [[] for _ in range(len(self.dataset.classes))]
    for img_idx, (_, class_id) in enumerate(self.dataset.samples):
      self.classes_to_img_indices[class_id].append(img_idx)
    
    # VisionDataset attributes for display
    self.root = root
    self.length = len(self.dataset.classes) # pseudo length! Length of this dataset is 7000, *not* the actual # of images in the dataset. You can just increase the # of epochs you train for.
    self.transforms = self.dataset.transforms
          
  def __len__(self):
    return self.length
    
  def __getitem__(self, anchor_class_idx):
    """Treat the given index as the anchor class and pick a triplet randomly"""
    anchor_class = self.classes_to_img_indices[anchor_class_idx]
    # choose positive pair (assuming each class has at least 2 images)
    anchor, positive = np.random.choice(a=anchor_class, size=2, replace=False)
    # choose negative image
    # hint for further exploration: you can choose 2 negative images to make it a Quadruplet Loss

    classes_to_choose_negative_class_from = list(range(self.length))
    classes_to_choose_negative_class_from.pop(anchor_class_idx) # TODO: What are we removing?
    negative_class = self.classes_to_img_indices[np.random.choice(a=classes_to_choose_negative_class_from, size=1)[0]] # TODO: How do we randomly choose a negative class?
    negative = np.random.choice(a=negative_class, size=1)[0] # TODO: How do we get a sample from that negative class?
    
    # self.dataset[idx] will return a tuple (image tensor, class label). You can use its outputs to train for classification alongside verification
    # If you do not want to train for classification, you can use self.dataset[idx][0] to get the image tensor
    return self.dataset[anchor], self.dataset[positive], self.dataset[negative]


# Dataset / Dataloader
class TripleData():
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
    
    def load(self, train=True, augment=True):
        # Augment images
        if train and augment:
            data_transform = data_aug
        else:
            data_transform = data_trans
        
        self.dataset = TripletDataset(self.dir, transform=ttf.Compose(data_transform))

        # Create data loader (train=shuffle)
        if train:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=True, drop_last=True, num_workers=2)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                            shuffle=False, drop_last=False, num_workers=2)
        
        # Find input size
        anchor, pos, neg = iter(self.data_loader).next()
        self.input_size = tuple(anchor[0].shape)

        return self.data_loader
    
    @staticmethod
    def find_n(dir, batch_size):
        data_transform = [ttf.ToTensor()]
        dataset = TripletDataset(dir, transform=ttf.Compose(data_transform))
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=True, num_workers=2)
        return len(dataset), len(data_loader)


class VerificationDataset(Dataset):
    def __init__(self, data_dir, transforms):
        
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # We return the image, as well as the path to that image (relative path)
        return self.transforms(Image.open(self.img_paths[idx])), osp.relpath(self.img_paths[idx], self.data_dir)


class VerificationData:
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
    
    def load(self):
        data_transform = data_trans
        self.dataset = VerificationDataset(self.dir, transforms=ttf.Compose(data_transform))
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=1)
        return self.data_loader
