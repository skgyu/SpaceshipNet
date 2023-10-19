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




class LMDBFeatureLabelImageDataset(Dataset):

    def __init__(self,db, do_norm_range=False ,num_data=-1 ):


        self.do_norm_range=do_norm_range

        self.env = db

        with self.env.begin(write=False) as txn:
            self.len =pickle5.loads(txn.get(b'__len__'))
            self.keys= pickle5.loads(txn.get(b'__keys__'))



        if num_data==-1:
            num_data= self.len


        assert( 0<num_data and num_data<= self.len )

        self.len=  num_data



    def __len__(self):

        return self.len



    def __getitem__(self, idx):
        

        idx=idx%self.len

        
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx].encode())
        feature, label, img = pickle5.loads(byteflow)

        feature=torch.Tensor(feature.copy()) 

        
        img=torch.Tensor(img.copy()) 

        label=copy.deepcopy(label)



        return   {'feature': feature , 'label':label , 'image':img}





def compare_lmdb(x):    

    a,b=x.split('-')
    a=int(a)
    b=int(b)
    return a,b






class LMDBFeatureLabelImageDataset0425(Dataset):

    def __init__(self,db, do_norm_range=False ,num_data=-1 ):


        self.do_norm_range=do_norm_range

        self.env = db

        with self.env.begin(write=False) as txn:
            self.len =pickle.loads(txn.get(b'__len__'))
            self.keys= pickle.loads(txn.get(b'__keys__'))



        if num_data==-1:
            num_data= self.len


        assert( 0<num_data and num_data<= self.len )

        self.len=  num_data


    def __len__(self):

        return self.len

    def update(self):

        with self.env.begin(write=False) as txn:
            self.len =pickle.loads(txn.get(b'__len__'))
            self.keys= pickle.loads(txn.get(b'__keys__'))

        self.txn= self.env.begin(write=False)

    def __getitem__(self, idx):
        

        idx=idx%self.len
        
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx].encode())
        feature, label, img = pickle.loads(byteflow)

        feature=torch.Tensor(feature.copy()) 
        
        img=torch.Tensor(img.copy()) 

        label=copy.deepcopy(label)


        return   {'feature': feature , 'label':label , 'image':img}



        







    


















