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
import networks,shutil
from skyu_tools import net_magnifier as smag
import torch.nn as nn
import functools
import torch.nn.functional as F
from torchvision import transforms
from kornia import augmentation
from datafree.utils import  DataIter
from timm.data.mixup import Mixup


class model_CFE_mSARC_base(BaseModel):
    def name(self):
        return 'model_CFE_mSARC_base'

    def init_loss(self):

        opt=self.opt
        BaseModel.init_loss(self)
        self.clear_loss()  



    def initialize(self, opt):


        BaseModel.initialize(self, opt)

        print('model_CFE_mSARC_base initialize')


        self.prin_sub_num =  torch.Tensor(opt['prin_normalize'][0]).view(1,3,1,1).contiguous().cuda()
        self.prin_div_num =  torch.Tensor(opt['prin_normalize'][1]).view(1,3,1,1).contiguous().cuda()
        

        sutil.add_functions(self,dirs='models',model_name='test_functions')
        
                
        if  hasattr(opt,'model_files'):  
            for  file in opt.model_files:
                sutil.add_functions(self,dirs=os.path.join('models',self.name()+'s'),model_name=file)
                shutil.copy(  os.path.join('models',self.name()+'s',file+'.py') , os.path.join(opt.expr_dir,file+'.py'))
        

        getattr(self,'initializeB')(opt)



    def update_learning_rate(self , epoch):


        BaseModel.update_learning_rate(self,epoch)


    def set_input(self,data):

        opt=self.opt


        self.inputs= data['image'].to(opt.device)
        self.labels= data['label'].to(opt.device)
        # self.poses= data['pos']



    def gradcam_cope(self,net,conv_outputs,model_output,one_hot_labels,retain_graph):

        
        grad = torch.autograd.grad(model_output, net.parameters(),grad_outputs =one_hot_labels,retain_graph=retain_graph)
        
        cams=[]
        
        for gradient,conv_output in zip(net.gradients[::-1],conv_outputs):

            weights = gradient.mean([2,3]) # Take averages for each gradient
            cams.append( (weights.unsqueeze(2).unsqueeze(3)* conv_output).sum(dim=1) )  

        return cams


    def init_optS(self):


        opt=self.opt

        self.optimizer_S  = torch.optim.SGD(self.student_net.parameters(), opt.optimizer_S.lr[0], weight_decay=opt.weight_decay, momentum=0.9)

    def initializeB(self,opt):


        sutil.log('model_CFE_mSARC_base','main')
        print('model_CFE_mSARC_base')


        ##################################################################################
        self.principle_names=['student_net']
        self.auxiliary_names=['generator','teacher_net']
        self.hiddens=[]
        self.optimizer_names=['optimizer_S']
        ##################################################################################


        self.student_net= networks.find_network(opt['student_net'])

        if opt.isTrain:
            self.teacher_net=  networks.find_network(opt['teacher_net'])
            self.teacher_net.eval()
     
            self.generator=     networks.find_network(opt['generator'])


        #################
        if not self.isTrain or opt.continue_train:  
            which_epoch = opt.which_epoch 
            self.load_network(self.student_net, 'student_net', which_epoch)
        else:  #####     
            print('#########train from scrach#############')
            self.student_net.apply(networks.weights_init(opt['student_net']['init'])  )
        #################

        if self.isTrain:
            if opt.continue_train:
                self.load_network(self.generator, 'generator', which_epoch)
            else:
                self.generator.apply(networks.weights_init(opt['generator']['init'])  )


        ###### GPU ###############
        for name in self.principle_names+self.auxiliary_names:
            if hasattr(self,name):
                getattr(self,name).to(self.opt.device)
        ###########################

        ####loss################
        if self.isTrain:
            self.init_loss()


            self.kl_loss_criterion =torch.nn.KLDivLoss(reduction='batchmean') 


            self.init_optS()
            
            if opt.continue_train:  
                self.load_optimizer(self.optimizer_G,'optimizer_G',opt.which_epoch )
                self.load_optimizer(self.optimizer_S,'optimizer_S',opt.which_epoch )


            self.init_scheduler()

            if opt.continue_train:  

                for i,name in enumerate(self.optimizer_names):

                    self.load_scheduler(self.schedulers[i],'scheduler_{}'.format(i), opt.which_epoch)


        ################


        self.n_row = min(opt.batch_size, 10)
        self.nrow =  3
        self.shownum= self.n_row*5


        self.sub_num =  torch.Tensor(opt['prin_normalize'][0]).view(1,3,1,1).contiguous().cuda()
        self.div_num =  torch.Tensor(opt['prin_normalize'][1]).view(1,3,1,1).contiguous().cuda()


        if self.isTrain:

            self.fc_like_conv = nn.Conv2d(512, opt.teacher_net.num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(opt.device)
            self.fc_like_conv.eval()

            self.loss_meanvar_feature_layers = []

            for module in self.teacher_net.modules():

                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                    self.loss_meanvar_feature_layers.append(smag.DeepInversionFeatureHook(module))




            self.aug =    transforms.Compose([ 
                    augmentation.RandomCrop(size=[opt.generator.img_size, opt.generator.img_size], padding=4),
                    augmentation.RandomHorizontalFlip()
                ])


            self.dataset_transform=  transforms.Compose([
                transforms.RandomCrop(opt.generator.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])


            self.gdataset_dir= os.path.join( opt.expr_dir,'synthesis')
            sutil.makedirs(self.gdataset_dir)



            
            if opt.use_mixup_cutmix:

                self.mixup_fn= Mixup(**opt.mixup_args)
        

        sutil.get_logger(path=os.path.join(opt.debug_info_dir,'metric_learning.log' ),name='metric' )

        sutil.log('input distance','metric')

    def normalize_before_cls(self,tensor):


        return (tensor-self.sub_num)/self.div_num



    def generate_input(self,test=False,labels= None):


        opt=self.opt

        num_classes=opt.teacher_net.num_classes

        if labels is None:

            if not test:
                labels=[]
                for i in range(opt.batch_size):
                    label= np.random.randint(num_classes)
                    labels.append(label)
            else:
                labels=[i for i in range(num_classes)]* (opt.batch_size//num_classes) + [i for i in range(opt.batch_size%num_classes)]


        codes=[]
        for label in labels:
            tmp=[0 for i in range(num_classes)]
            tmp[label]=1
            codes.append(tmp)

        noise= torch.randn( (opt.batch_size, opt.input_dim-num_classes) ).cuda()

        code = torch.FloatTensor(codes).cuda().contiguous()

        self.labels=   torch.LongTensor(labels).cuda()

        return torch.cat([noise,code],dim=1)


    def reset_model(self,model):
        for m in model.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)) or isinstance(m, nn.SyncBatchNorm):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


    def train_G(self):

        pass


    def cope_fclikeconv(self,net,conv_output):
        
        pass


    def train_S(self):

        pass


    def optimize_step(self):


        assert(self.student_net.training)

        self.train_G()

        self.train_S()


    def trainstate(self):
        
        self.student_net.train()
        self.generator.train()


    def evalstate(self):

        self.student_net.eval()
        self.generator.eval()

        if hasattr(self,'teacher_net'):
            self.teacher_net.eval()


    def test(self,epoch,phase):

        n_row =  self.n_row
        nrow =  2*self.nrow


        nimg=len( glob.glob( os.path.join(self.opt.debug_image_dir,'sample_{}*'.format(phase) )  ))
        typename=    'epoch{}_{}'.format(epoch,nimg) 
        sutil.log('type='+typename,'sample')


        opt=self.opt

        z = torch.randn(size=(opt.batch_size, opt.generator.input_dim), device=opt.device)


        with torch.no_grad():
            fake_data=self.generator(z)[0]
            fake_data=fake_data.detach()

        images=[]
        images.append( fake_data[0:self.shownum])
      
      
        tensor_image=torch.cat(images,dim=0)
        image_grid=torchvision.utils.make_grid(tensor_image, nrow=n_row, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
        image_numpy=sutil.tensor2im(image_grid, cent=0, factor=255)
        sutil.save_image( image_numpy,  os.path.join(   self.opt.debug_image_dir  ,'sample_{}_{}.png'.format(phase,typename)   ) )



    def clear_loss(self):

        self.sum_loss= OrderedDict()
        self.cnt_loss= OrderedDict()
        self.accuracy_forG= 0
        self.sizes_C_forG = 0


        self.accuracy_forS= 0
        self.sizes_C_forS = 0


        self.correct_forG_last_iter= -1
        self.batch_size_forG_last_iter= -1
