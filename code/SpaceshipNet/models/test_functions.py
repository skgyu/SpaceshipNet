#-*- coding:utf-8 -*-
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from PIL import Image
import numpy as np
import os,time,ntpath,imp,logging,sys,random,copy,importlib
import pandas as pd
from torch.nn import DataParallel as DPL
from skyu_tools import skyu_util as sutil
import networks
import glob,torchvision


def testC(self,epoch):


    if hasattr(self,'sizes_C_forG') and self.sizes_C_forG!=0:
       
        sutil.log('epoch:='+ str(epoch),'cls.log'  )
        sutil.log('class_accuracy_forG:='+ str(1.0*self.accuracy_forG/self.sizes_C_forG),'cls.log'  )    

    if hasattr(self,'sizes_C_forS') and self.sizes_C_forS!=0:
       
        sutil.log('epoch:='+ str(epoch),'cls.log'  )
        sutil.log('class_accuracy_forS:='+ str(1.0*self.accuracy_forS/self.sizes_C_forS),'cls.log'  )    

    if hasattr(self,'correct_forG_last_iter') and self.correct_forG_last_iter!=-1:
       
        sutil.log('epoch:='+ str(epoch),'cls.log'  )
        sutil.log('predict_correct_num  forG_last_iter:= {}/{}'.format(self.correct_forG_last_iter, self.batch_size_forG_last_iter) ,'cls.log'  )




def record_batchimg(self,fake_data,epoch,best_epoch=None,pre='',save_per_image=False,labels=None):
    

    epoch=str(epoch)

    n_row =  self.n_row
    nrow =  2*self.nrow


    debug_image_dir= os.path.join(self.opt.debug_image_dir,epoch)
    sutil.makedirs(debug_image_dir)

    nimg=len( glob.glob( os.path.join(debug_image_dir,'sample_*' )  ))
    typename=    pre
    sutil.log('type='+typename,'sample')


    opt=self.opt



    if labels is None:
        labels=self.labels.cpu().numpy()

    fake_data=fake_data.detach()

    images=[]

    images.append( fake_data)
  
  
    tensor_image=torch.cat(images,dim=0)




    image_grid=torchvision.utils.make_grid(tensor_image, nrow=n_row, padding=0, normalize=True, range=None, scale_each=True, pad_value=0)
    image_grid=image_grid*2-1
    image_numpy=sutil.tensor2im(image_grid)

    sutil.save_image( image_numpy,  os.path.join(   debug_image_dir  ,'sample_{}.png'.format(typename)   ) )


    txtloc = os.path.join(   debug_image_dir  ,'sample_{}_labels.txt'.format(typename)   )

    with open(txtloc,'w') as fw:

        for st in range(0,opt.batch_size,self.n_row):
            ed=st+self.n_row

            ed=min(ed,opt.batch_size)
            for i in range(st,ed):
                fw.write('{} '.format(labels[i]) )
            fw.write('\n')

        if best_epoch:
            fw.write('best_epoch = {}\n'.format(best_epoch))

        fw.write('cated tensor_image\n')        
        fw.write('min max:\n')        
        fw.write('{} {}\n'.format( torch.min(tensor_image).item(), torch.max(tensor_image).item()  ) )

        fw.write('image_grid after*2 -1(normalize=True, range=None, scale_each=True)\n')        
        fw.write('min max:\n')        
        fw.write('{} {}\n'.format( torch.min(image_grid).item(), torch.max(image_grid).item()  ) )



        fw.write('cated 0th image of fake_data i.e., fake_data[0:1]\n')        
        fw.write('min max:\n')        
        fw.write('{} {}\n'.format( torch.min(fake_data[0:1]).item(), torch.max(fake_data[0:1]).item()  ) )


        image_grid_test=torchvision.utils.make_grid(fake_data[0:1], nrow=n_row, padding=0, normalize=True, range=None, scale_each=True, pad_value=0)

        fw.write('image_grid_test (normalize=True, range=None, scale_each=True)\n')        
        fw.write('min max:\n')     
        fw.write('{} {}\n'.format( torch.min(image_grid_test).item(), torch.max(image_grid_test).item()  ) )
   


    fw.close()

    torchvision.utils.save_image(fake_data.clone(),
                          os.path.join(   debug_image_dir  ,'sample_{}_save_image.png'.format(typename)   ) ,
                          normalize=True, scale_each=True, nrow=10)

    if save_per_image:

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range=None):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))



        savedir=os.path.join(   debug_image_dir, pre+'_imgs')
        sutil.makedirs( savedir  )

        for i in range(20):

            tmp_img=fake_data[i].clone().detach()

            norm_range(tmp_img)
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)

            
            saveloc=os.path.join(   savedir ,'sample_%02d.png'%i   )

            sutil.save_image( image_numpy,  saveloc )



def predict(self,test_net_name='student_net'):

    test_net = getattr(self,test_net_name)

    with torch.no_grad():

        prep_input=(self.inputs-self.prin_sub_num)/self.prin_div_num

        return test_net(prep_input)[0]