# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL
'''
edited from resnet_cifar_1028.py 
for adapt for tiny-imagenet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks




class ResNet(networks.resnet_cifar.ResNet):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__(block=block, num_blocks=num_blocks, num_classes=num_classes)

        
        self.bool_save_gradient=False
        
        target_layers=[]
        
        target_layers.append( len(self.layer1._modules)-1)
        target_layers.append( len(self.layer2._modules)-1)
        target_layers.append( len(self.layer3._modules)-1)
        target_layers.append( len(self.layer4._modules)-1)
        
        self.target_layers=target_layers
        self.linear_attr = 'linear'
        self.linear_input_dim= 512*block.expansion

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

        x = self.bn1(x)
        x = F.relu(x)

        #这个conv_output_l1是layer1最后一个残差块的输出  即 F.relu(block(x)+shortcut(x))
        for module_pos, module in self.layer1._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[0]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient) 
                conv_output_l1 = x  # Save the convolution output on that layer
                    
        for module_pos, module in self.layer2._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[1]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient)   
                conv_output_l2 = x  # Save the convolution output on that layer
        
        for module_pos, module in self.layer3._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[2]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient)
                conv_output_l3 = x  # Save the convolution output on that layer
                
        
        for module_pos, module in self.layer4._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layers[3]:
                if self.bool_save_gradient:
                    x.register_hook(self.save_gradient)
                conv_output_l4 = x  # Save the convolution output on that layer
                
        #conv_output = self.layer4(x)

        cam_feature=x
        # x = F.avg_pool2d(x, 4)
        x = F.adaptive_avg_pool2d(cam_feature, (1,1))

        feature = x.view(x.size(0), -1)
        out = self.linear(feature)
        
        return out,feature,cam_feature,[conv_output_l1,conv_output_l2,conv_output_l3,conv_output_l4]
 
 
def ResNet18_1028_aap(num_classes=10):
    return ResNet(networks.resnet_cifar.BasicBlock, [2,2,2,2], num_classes)
 
def ResNet34_1028_aap(num_classes=10):
    return ResNet(networks.resnet_cifar.BasicBlock, [3,4,6,3], num_classes)
 
def ResNet50_1028_aap(num_classes=10):
    return ResNet(networks.resnet_cifar.Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101_1028_aap(num_classes=10):
    return ResNet(networks.resnet_cifar.Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152_1028_aap(num_classes=10):
    return ResNet(networks.resnet_cifar.Bottleneck, [3,8,36,3], num_classes)
 


