# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import math



def get_scheduler(optimizer, optimizer_name, opt  ):

    if 'continue_train' not in opt or not opt.continue_train:
        last_epoch=-1

    elif isinstance(opt.which_epoch,int):
        last_epoch=int(opt.which_epoch)
        for i,x in enumerate(optimizer.param_groups): 
            x['initial_lr']=opt[optimizer_name]['lr'][i]
    else:
        raise(RuntimeError('opt.which_epoch??') )

    lr_strategy =  opt[optimizer_name]
    lr_policy = lr_strategy['lr_policy']




    if lr_policy == 'lambda2':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch  + 1 - lr_strategy.niter) / float(lr_strategy.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch) 

    elif lr_policy == 'lambda2_min':

        min_lr=lr_strategy['min_lr']

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch  + 1 - lr_strategy.niter) / float(lr_strategy.niter_decay + 1)

            if lr_l<min_lr:
                lr_l=min_lr
            
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)  

    elif lr_policy == 'origin':
        def lambda_rule(epoch):
            lr_l = 1
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch ) 

    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_strategy.step_size, gamma=lr_strategy.gamma , last_epoch=last_epoch )
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'CosineAnnealingLR':
        T_max =  lr_strategy['CosineAnnealingLR_T_max'] if 'CosineAnnealingLR_T_max' in lr_strategy else opt.max_epoch+1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=T_max, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    
    print('lr={}'.format(optimizer.param_groups[0]['lr']) )

    return scheduler




def get_scheduler_B(optimizer, optimizer_name, opt , max_iter, last_iter  ):

    lr_strategy =  opt[optimizer_name]
    lr_policy = lr_strategy['lr_policy']
   
    if lr_policy == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=max_iter+1, last_epoch=last_iter)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)

    return scheduler






