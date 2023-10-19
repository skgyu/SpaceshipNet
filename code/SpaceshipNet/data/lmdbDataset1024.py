#coding: utf-8
import numpy as np
import os
from torch.utils.data import Dataset
import random
import cv2
import torch
import pickle
import torchvision.transforms as transforms
import time
import os.path as osp
import pyarrow as pa
import lmdb
import copy
import pickle5
import pickle
from PIL import Image



def norm_ip_(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range_(t, value_range=None):
    if value_range is not None:
        norm_ip_(t, value_range[0], value_range[1])
    else:
        norm_ip_(t, float(t.min()), float(t.max()))






class LMDBFeatureLabelImageDataset1024(Dataset):

    def __init__(self,db, do_norm_range=False ,num_data=-1 ):


        self.do_norm_range=do_norm_range


        self.len =db['__len__']
        self.keys= db['__keys__']


        self.labels =  db['labels']
        self.images = db['images']
        self.features = db['features']

        if num_data==-1:
            num_data= self.len

        assert( 0<num_data and num_data<= self.len )

        self.len=  num_data



    def __len__(self):

        return self.len



    def __getitem__(self, idx):
        

        idx=idx%self.len


        savename = self.keys[idx]

        iepoch, epoch_idx =  savename.split('-')
        iepoch = int(iepoch)
        epoch_idx = int(epoch_idx)


        label = self.labels[iepoch][epoch_idx]

        img = self.images[iepoch][epoch_idx]

        feature = self.features[iepoch][epoch_idx]        
        
        feature=torch.Tensor(feature) 
        img=torch.Tensor(img) 


        return   {'feature': feature , 'label':label , 'image':img}



def compare_lmdb(x):    

    a,b=x.split('-')
    a=int(a)
    b=int(b)
    return a,b









        







    


















