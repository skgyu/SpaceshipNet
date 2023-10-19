# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import math
###############################################################################
# Functions
###############################################################################


def reset_bn_running_state(m):
    classname = m.__class__.__name__
    global idx
    
    if classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        #print(m.running_mean)
        #print(m.running_var)
        m.reset_running_stats()
        

        
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    return init_fun

def weights_init_zero(m):

    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0,0.0)
        if m.bias  is not None:
            init.normal_(m.bias.data, 0.0 ,0.0)

    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0 ,0.0)
        if m.bias  is not None:
            init.normal_(m.bias.data, 0.0 ,0.0)

    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.normal_(m.weight.data, 0.0 ,0.0 )
        init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data)# gain default=1
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1: 
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'xavier_uniform':
        net.apply(weights_init_xavier_uniform)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'zero':
        net.apply(weights_init_zero)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)





