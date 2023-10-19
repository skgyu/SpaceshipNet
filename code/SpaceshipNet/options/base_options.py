## coding:utf-8
import os
import argparse
import torch
import time
import yaml
from easydict import EasyDict as edict
import shutil
from skyu_tools import  skyu_util as sutil
import random
import numpy as np
import torchvision.transforms as transforms
import torch.distributed as dist






class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--datasets', type=str, help='')
        self.parser.add_argument('--config', type=str, help='Path to the config file.')
        self.parser.add_argument('--test_fre', default=1, type=int, help='test_fre')
        self.parser.add_argument('--exps_dir', default='./experiments', type=str, help='exps_dir')
        self.parser.add_argument('--batch_size',  default=4, type=int)
        self.parser.add_argument("--local_rank", type=int, default=0, help="")
        self.parser.add_argument("--FeatureLabelImagePoolType", type = str, default= 'SkyuFeatureLabelImagePool', help='SkyuFeatureLabelImagePool|SkyuFeatureLabelImagePool1024' )
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt = edict(vars(opt))
        opt.isTrain = self.isTrain   # train or test

        config_addrs= opt.config.split(',')

        for config_addr in config_addrs:
            config_addr=config_addr.strip()

            with open(config_addr, 'r') as stream:
                #config =yaml.load(stream)
                config =yaml.load(stream, Loader=yaml.SafeLoader)
                
                for key,value in  config.items():        
                    opt[key]=value

        gpus=''
        for i in range(len(opt.gpu_ids)):
            gpus+= str(opt.gpu_ids[i])
            if i!= len( opt.gpu_ids)-1:
                gpus+=','

        print(gpus)
        os.environ['CUDA_VISIBLE_DEVICES']=gpus


        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(0)
            opt.device_str="cuda:0" if torch.cuda.is_available() else "cpu"
            opt.device = torch.device( opt.device_str ) 

        else:
            opt.device= torch.device('cpu')


        opt.gpu_ids=   [i  for i in range(len(opt.gpu_ids)) ]



        ################dataset############################
        if hasattr(opt,'datasetids'):
            opt.datasetids.sort()


        if hasattr(opt,'test_dataset_ids'):
            opt.test_dataset_ids.sort()

        if hasattr(opt,'test_source_dataset_ids'):
            opt.test_source_dataset_ids.sort()


        expr_dir = os.path.join(opt.exps_dir, opt.name) 



        if opt.local_rank==0:
            sutil.makedirs(expr_dir)        

        file_name = os.path.join(expr_dir, 'opt.txt')
        opt.expr_dir=expr_dir





        if opt.local_rank==0:
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(opt.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')


        

        config_name = 'config_train' if self.isTrain  else 'config_test'
        config_dir=os.path.join(expr_dir, config_name)
        cnt=1
        while os.path.exists(config_dir):
            config_dir=os.path.join(expr_dir, config_name + str(cnt))
            cnt+=1


        if opt.local_rank==0:
            sutil.makedirs(config_dir)


            for config_addr in config_addrs:
                shortname= os.path.split(config_addr)[-1]
                shutil.copy(config_addr,   os.path.join(config_dir,shortname)   ) # copy config file to output folder


        opt.debug_image_dir= os.path.join( expr_dir,'rank{}_image'.format(opt.local_rank) )
        opt.debug_info_dir= os.path.join( expr_dir,'rank{}_info'.format(opt.local_rank) )
        opt.grad_info_dir= os.path.join( expr_dir,'rank{}_grad'.format(opt.local_rank) )
        

        sutil.makedirs(opt.debug_image_dir)
        sutil.makedirs(opt.debug_info_dir)
        sutil.makedirs(opt.grad_info_dir)


        sutil.get_logger(path=os.path.join(opt.debug_info_dir,opt.name+'.log' ),name='main' )
        sutil.get_logger(path=os.path.join(opt.grad_info_dir ,'grad.log' ),name='grad' )


        if opt.local_rank==0:
            sutil.get_logger(path=os.path.join(config_dir ,'config_edit.log' ),name='config' )
            sutil.get_logger(path=os.path.join(opt.expr_dir ,'datasetids.log' ),name='datasetids' )


        if opt.isTrain: 
            print(opt.batch_size)


            
        def to255(x):

            return (x+1)/2.0*255.0



        for normalize_name in ['train_normalize','prin_normalize']:
            if normalize_name in opt:
                if len(opt[normalize_name][0])==3:
                    opt[normalize_name+'_func']=transforms.Normalize(  mean=tuple(opt[normalize_name][0]), std=tuple(opt[normalize_name][1]) )
                elif len(opt[normalize_name][0])==1:
                    opt[normalize_name+'_func']=transforms.Normalize(  mean=opt[normalize_name][0], std=opt[normalize_name][1] )
                else:
                    raise(RuntimeError('train_normalize is not correct'))


        if opt.isTrain and opt.continue_train and  isinstance(opt.which_epoch,int):
            opt.start_epoch=int(opt.which_epoch)



        if opt.name.find('_Cifar10_')!=-1:
            opt.dataset='cifar-10'
        elif opt.name.find('_Cifar100_')!=-1:
            opt.dataset='cifar-100'
        elif opt.name.find('tiny_imagenet')!=-1:
            opt.dataset='tiny_imagenet'        
        elif opt.name.find('PACS')!=-1:
            opt.dataset='PACS'   
        elif opt.name.find('imagenette2')!=-1:
            opt.dataset='imagenette2'
            
        elif opt.name.find('imagenet100')!=-1:
            opt.dataset='imagenet100'
        else:
            raise(RuntimeError('unknown dataset'))      

        self.opt = opt

        sutil.upload_opt(opt)

        assert('manualSeed' in opt)

        random.seed(opt.manualSeed)
        os.environ['PYTHONHASHSEED'] = str(opt.manualSeed)
        np.random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)    
        torch.backends.cudnn.benchmark = False            # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = True

        return self.opt



       