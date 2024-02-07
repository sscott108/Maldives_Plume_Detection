import os
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
from math import floor ,log10
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import roc_curve, confusion_matrix
import warnings
import torch.nn.functional as nnf
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio as rio
import time 
from torchvision.models import resnet50
warnings.filterwarnings("ignore")
torch.manual_seed(1865)
from sklearn.model_selection import train_test_split
import torch.nn as nn

class SmokePlumeDataset():
    def __init__(self, datadir=None, mult=1, transform=None, balance='upsample'):
        self.datadir = datadir
        self.transform = transform
        self.imgfiles = []  # list of image files
        self.labels = []    # list of image file labels
        self.positive_indices = []  # list of indices for positive examples
        self.negative_indices = []  # list of indices for negative examples

        # read in image file names
        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.jpg'):
                    continue
                self.imgfiles.append(os.path.join(root, filename))
                if 'positive' in root:
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    idx += 1
                elif 'negative' in root:
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    idx += 1

        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        """Read in image data, preprocess, and apply transformations."""
        # read in data file
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in [1,2,3]])

        sample = {
            'idx': idx,
            'img': imgdata,
            'lbl': self.labels[idx],
            'imgfile': self.imgfiles[idx]
        }

        return sample

    def display(self, idx, offset=0.2, scaling=1.5):
        imgdata = self[idx]['img']

        # scale image data
        imgdata = offset + scaling * (
            np.dstack([imgdata[3], imgdata[2], imgdata[1]]) -
            np.min([imgdata[3], imgdata[2], imgdata[1]])) / \
                (np.max([imgdata[3], imgdata[2], imgdata[1]]) -
                 np.min([imgdata[3], imgdata[2], imgdata[1]]))

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow((imgdata - np.min(imgdata, axis=(0, 1))) /
                  (np.max(imgdata, axis=(0, 1)) -
                   np.min(imgdata, axis=(0, 1))))

        return f

class ToTensor:
    def __call__(self, sample):

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'lbl': sample['lbl'],
               'imgfile': sample['imgfile']}
        return out

class Normalize:
    def __init__(self):
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):

        sample['img'] = (sample['img'] - self.channel_means.reshape(
            sample['img'].shape[0], 1, 1)) / self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample


class Randomize:

    def __call__(self, sample):

        imgdata = sample['img']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1, 2))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}

class RandomCrop:

    def __call__(self, sample):
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}

def create_dataset(*args, apply_transforms=True, **kwargs):
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            ToTensor(),
        ])
    else:
        data_transforms = None

    data = SmokePlumeDataset(*args, **kwargs, transform=data_transforms)

    return data
