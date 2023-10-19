#coding: utf-8
import numpy as np
import os
from torch.utils.data import Dataset
import random
import cv2
import torch
import pickle
import torchvision.transforms as transforms




def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class numpyimage_to_tensor:

    def __init__(self,img_size):

        self.img_size = img_size


    def __call__(self,img):


        img=np.transpose(img,(2,0,1))
            
        img= img/255.0

        img=torch.from_numpy(img).float()

        return img




class Cifar10Dataset(Dataset):

    def __init__(self,root,train,aug,normalize_transfer=None ):


        self.dataroot=  os.path.join(root,'cifar-10-batches-py')

        data=[]

        labels=[]

        self.img_size=32

        self.aug=aug

        if not train:

            self.aug=False

            test_batch='test_batch'

            test_batch_loc= os.path.join(self.dataroot,test_batch)

            dic=unpickle(test_batch_loc)

            data.append(dic['data'.encode()])

            labels+=dic['labels'.encode()]


        else:

            train_batches=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']


            for train_batch in train_batches:

                train_batch_loc=os.path.join(self.dataroot,train_batch)

                dic=unpickle(train_batch_loc)

                data.append(dic['data'.encode()])

                labels+=dic['labels'.encode()]



        data = np.concatenate(data,axis=0)

        self.data = data.reshape(-1,3,self.img_size,self.img_size).transpose(0,2,3,1)

        self.len_imageX= data.shape[0]

        self.labels=labels

        assert(self.len_imageX==len(self.labels))

        self.numpyimage_to_tensor = numpyimage_to_tensor(self.img_size)


    def __len__(self):

        return self.len_imageX


    def augX_opencv(self,img_x):


        pflip, pcrop  = random.uniform(0, 1),random.uniform(0, 1)

        if pflip>0.5:

        
            img_x= img_x[:,::-1]


        if pcrop>0.5:

            
            img_x=cv2.resize(img_x, (self.img_size+4, self.img_size+4), interpolation=cv2.INTER_LINEAR)

            starth= random.randint(0,4)
            startw= random.randint(0,4)

            img_x=img_x[starth:starth+self.img_size,startw:startw+self.img_size]

        return img_x


    def __getitem__(self, idx):
        

        idx_imageX=idx%self.len_imageX

        imageX=self.data[idx]

        if self.aug:
            imageX = self.augX_opencv(imageX)


        imageX = self.numpyimage_to_tensor(imageX)

        imageX_label =   self.labels[ idx_imageX ]

        return   {'image': imageX, 'label': imageX_label, 'pos': idx_imageX}







        







    


















