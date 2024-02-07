#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from os import listdir
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.optim import Adam
import os
import json
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import jaccard_score
import cv2
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore')
from PIL import Image
# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"
import datetime
today = datetime.date.today()
import time
import torchvision.models.segmentation
time = datetime.datetime.now()
from sklearn.model_selection import KFold
from scipy.special import softmax
from torch.utils.data import Dataset, ConcatDataset
import pickle as pkl
seed= np.random.randint(0,10000)
torch.manual_seed(seed)
from sklearn.model_selection import train_test_split
import torchmetrics



if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 

model_root_dir = '.'
model_save_dir = 'saved_models'

if not (os.path.isdir(model_root_dir + '/' + model_save_dir)):
    print('Model saving directory didn''t exist, creating now.')
    os.mkdir(model_root_dir+'/'+model_save_dir)
    
else:
    print('Model saving directory exists.')



def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)
                


def pixel_accuracy(output, label):
    output = torch.argmax(output, dim=1)
    correct = torch.eq(output, label).float()
    accuracy = torch.mean(correct)
#     print(output.sum())
#     if output.sum() >= 0.5:
#          print('positive')
#     else:
#         print('negative')
    return accuracy

# def pixel_accuracy(output, mask):
#     with torch.no_grad():
#         output = torch.argmax(F.softmax(output, dim=1), dim=1)
#         correct = torch.eq(output, mask).int()
#         accuracy = float(correct.sum()) / float(correct.numel())
#     return accuracy
 

# def pixel_accuracy(output, mask):
#     output = np.argmax(softmax(output, axis=1), axis=1)
#     correct = (output == mask)
#     accuracy = np.sum(correct) / correct.size
#     return accuracy


# def weighted_focal_loss(outputs, targets, alpha, gamma):
#     alpha = torch.tensor([alpha, 1-alpha])
#     gamma = gamma
#     BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none').squeeze(dim=1).flatten()
#     targets = targets.type(torch.long)
#     at = alpha.gather(0, targets.data.view(-1))
#     pt = torch.exp(-BCE_loss).flatten()
#     F_loss = at*(1-pt)**gamma * BCE_loss
#     return F_loss.mean()

def image_comparison(ogimg, ogmask, prediction,iou, save = True, show = True, train_o_test='', fig_name =''):
    
    f, axs = plt.subplots(1, 3, figsize=(10, 7))

    axs[0].imshow(ogimg.permute(1, 2, 0))        #original image, no labels
    axs[0].set_title('Original '+train_o_test+' Image')
    axs[0].axis('off')

    axs[1].imshow(ogmask, cmap='gray')           #ground truth label
    axs[1].set_title('Ground Truth Label')
    axs[1].axis('off')

    axs[2].imshow(prediction, cmap= 'gray')                      #prediction
    axs[2].set_title(train_o_test +' Prediction')
    axs[2].axis('off')
    axs[2].text(0.5, -0.1, f'IoU:{iou}') #, transform=axs[2].transAxes,
            #bbox=dict(facecolor='white', alpha=0.6, color='black', fontsize=10))    
    if save== True:
        f.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'_'+'.png', transparent=False, facecolor='white', bbox_inches='tight')
    if show == True:
        plt.show()
        
    plt.close() 
    
    
def train_test_loss(loss_train, loss_test, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (6,6))   
    ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Training')
    ax.plot(epoch, loss_test, color='r', linewidth=0.5, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')

def train_test_ious(train_ious, test_ious, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (6,6))   
    ax.plot(epoch, train_ious, color='b', linewidth=0.5, label='Training')
    ax.plot(epoch, test_ious, color='r', linewidth=0.5, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('IoU')
    ax.set_title('Training and Validation IoU')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')

def train_test_acc(train_accuracy, test_accuracy, epochs, save = True, fig_name=''):
    epoch = range(epochs)
    fig, ax = plt.subplots(1,1, figsize = (6,6))   
    ax.plot(epoch, train_accuracy, color='b', linewidth=0.5, label='Training')
    ax.plot(epoch, test_accuracy, color='r', linewidth=0.5, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    plt.show()
    if save==True:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')     
        
def total_v_pred(df1, df2, epoch):
    plt.figure(figsize=(7,7))
    sns.set_style("darkgrid")
    conc = pd.concat([df1, df2]).reset_index(drop=True)
    svm = sns.scatterplot(data=conc, x="Truth", y="Predicted", hue='set')
    svm.set_title("Predicted Plume Pixels vs. Total Plume Pixels")
    plt.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+'total_v_pred_'+str(epoch)+'.png', transparent=False, facebplor='white',bbox_inches='tight')
#     plt.show()
    plt.close()
    
class PositiveOnly(Dataset):
    def __init__(self, p_pkl, transform=None):
        self.p_pkl = p_pkl
        self.transform = transform
        
        with open(p_pkl, "rb") as fp:
            self.p_pkl = pkl.load(fp)
        
    def __len__(self):
        return len(self.p_pkl)
    
    def __getitem__(self, idx):
        sample = self.p_pkl[idx]
#         print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def display(self, idx):

        sample = self[idx]
        
        imgdata = sample['img'] #.permute(1, 2, 0)
        fptdata = sample['fpt']


        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(imgdata.T)
        ax[0].imshow(fptdata.T, alpha = 0.5)
        ax[1].imshow(imgdata.T)
        ax[1].axis('off')
        ax[0].axis('off')
        # plt.show()
        return f   
    
class Randomize(object):
    def __call__(self, sample):

        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
#         rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))
        fptdata = np.rot90(fptdata, rot, axes=(0,1))
        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}
        return out
    
def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Randomize(),
            ToTensor(),
        ])
    else:
        data_transforms = None

    data = PositiveOnly(*args, **kwargs, transform=data_transforms)
    return data

class NegativeOnly():
    def __init__(self, img_dir = None, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.imgfiles = []
        self.seglabels = []
        
        for f in listdir(img_dir):
            if f.endswith('.jpg'):
                self.imgfiles.append(os.path.join(img_dir,f))
                self.seglabels.append([])

        self.imgfiles = np.array(self.imgfiles)
    def __len__(self): return len(self.imgfiles)
    
    def __getitem__(self, idx):
        
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                           [1,2,3]])
        fptdata = np.zeros(imgdata.shape[1:])      #creates 288x288 array of zeros

        
        sample = {'idx': idx,
                  'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgfiles[idx]}
        
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    def display(self, idx):

        sample = self[idx]
        
        imgdata = sample['img'] #.permute(1, 2, 0)
        fptdata = sample['fpt']


        f, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(imgdata.T)
        ax[0].imshow(fptdata.T, alpha = 0.5)
        ax[1].imshow(imgdata.T)
        ax[1].axis('off')
        ax[0].axis('off')
        # plt.show()
        return f 
    
def create_dataset_n(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Randomize(),
            ToTensor(),
        ])
    else:
        data_transforms = None

    data = NegativeOnly(*args, **kwargs, transform=data_transforms)
    return data
