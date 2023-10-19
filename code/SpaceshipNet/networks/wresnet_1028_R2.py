'''https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/wresnet.py 
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
import time



__all__ = ['wrn']



class WideResNet(networks.wresnet.WideResNet):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__( depth =depth, num_classes= num_classes, widen_factor= widen_factor, dropout_rate= dropout_rate)

        self.bool_save_gradient=False
        
        target_layers=[]
        
        target_layers.append( len(self.block1._modules)-1)
        target_layers.append( len(self.block2._modules)-1)
        target_layers.append( len(self.block3._modules)-1)
        
        print(target_layers)
        self.target_layers=target_layers
        self.linear_attr = 'fc'
        self.linear_input_dim= self.nChannels


    def do_save_gradient(self,bool_save_gradient=True):

        self.bool_save_gradient=bool_save_gradient

        if not bool_save_gradient:
            self.gradients=[]

    def save_gradient(self, grad):

        self.gradients.append(grad)

    def get_gradient(self):

        return self.gradients

    def forward(self, x):
        
        if self.bool_save_gradient:
            self.gradients=[]
        
        x = self.conv1(x)

        
        for module_pos, module in self.block1.layer._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[0]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient) 
                conv_output_l1 = x  # Save the convolution output on that layer
                    
        for module_pos, module in self.block2.layer._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[1]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient)   
                conv_output_l2 = x  # Save the convolution output on that layer
        
        for module_pos, module in self.block3.layer._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[2]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient)
                conv_output_l3 = x  # Save the convolution output on that layer
                
        
        cam_feature = self.relu(self.bn1(x))
        
        x = F.adaptive_avg_pool2d(cam_feature, (1,1))

        feature = x.view(-1, self.nChannels)

        out = self.fc(feature)

        return out,feature,cam_feature,[conv_output_l1,conv_output_l2,conv_output_l3]
    
 

def wrn_16_1_1028_R2(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_16_2_1028_R2(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)

def wrn_40_1_1028_R2(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_40_2_1028_R2(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)




'''
resnet 

size 32
->layer2 16
->layer3  8
->layer4  4
->  out = F.avg_pool2d(out, 4)  1


block1
torch.Size([6, 32, 32, 32])

block2
torch.Size([6, 64, 16, 16])

block3
torch.Size([6, 128, 8, 8])

out = F.adaptive_avg_pool2d(out, (1,1))
torch.Size([6, 128, 1, 1])

cam:  bs 10 8  8

showgradcams
torch.Size([bs, 32, 32])
torch.Size([bs, 16, 16])
torch.Size([bs, 8, 8])
'''