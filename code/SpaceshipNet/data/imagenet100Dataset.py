#coding: utf-8
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import h5py
from PIL import Image
import json


class numpyimage_to_tensor:

    def __init__(self,img_size):

        self.img_size = img_size


    def __call__(self,img):
            
        img= img/255.0

        img=torch.from_numpy(img).float()

        return img



class Imagenet100Dataset(Dataset):

    def __init__(self,root,phase,img_size, aug=False, pad_size=15, use_resize=True, RS_transfer_func='RS_transfer'):


        self.dataroot=  os.path.join(root,'imagenet100')


        folder_name2label={}

        with open(os.path.join(self.dataroot,'Labels.json'), 'r', encoding='utf-8') as fw:
            classdict = json.load(fw)
            class_numbers=list(classdict.keys())
            for class_idx, class_number in enumerate(class_numbers) :
                folder_name2label[class_number] = class_idx


        assert( isinstance(phase,str ) )
        self.img_size = img_size
        self.pad_size = pad_size
        self.aug=aug


        image_paths=[]
        labels=[]
        all_datafolders = os.listdir(self.dataroot)

        for tmp_datafolder in all_datafolders:

            if tmp_datafolder.find(phase)==-1:
                continue

            datafolder_loc = os.path.join(self.dataroot, tmp_datafolder)
            folders = os.listdir(datafolder_loc)

            for folder in folders:
                folder_loc = os.path.join(datafolder_loc,folder)    
                files = os.listdir(folder_loc)
                files.sort()
                for file in files:

                    image_paths.append(os.path.join(folder_loc,  file)  )
                    labels.append(folder_name2label[folder])

          
        self.len_image= len(image_paths)

        self.image_paths=image_paths
        self.labels=labels
        assert(self.len_image==len(self.labels))

        if phase!='train':
            assert(aug==False)

        if phase=='train':

            if self.aug:
                func=getattr(self,RS_transfer_func)
                self.PILtoTensor_func= func(img_size,img_size,3)
            else:
                self.PILtoTensor_func= self.normal_transfer(3)

        else:

            if use_resize:
                self.PILtoTensor_func= self.normal_transfer(3)
            else:
                self.PILtoTensor_func= self.test_transfer(3)





    def __len__(self):

        return self.len_image


    def RS_transfer(self,img_size_H,img_size_W,nchannel):

        return transforms.Compose([
            transforms.Resize((img_size_H , img_size_W)),
            transforms.RandomCrop(self.img_size, padding=self.pad_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def normal_transfer(self, nchannel):

        return  transforms.Compose([
            transforms.Resize( (self.img_size , self.img_size) ),
            transforms.ToTensor()
        ])    

    def test_transfer(self, nchannel):

        return  transforms.Compose([
            transforms.CenterCrop( (self.img_size , self.img_size) ),
            transforms.ToTensor()
        ])



    def __getitem__(self, idx):
        

        idx_image = idx % self.len_image

        imageX= Image.open(self.image_paths[idx_image]).convert('RGB')


        imageX = self.PILtoTensor_func(imageX)

        imageX_label =   self.labels[ idx_image ]


        return   {'image': imageX, 'label': imageX_label}






    


















