"""https://github.com/HobbitLong/RepDistiller/blob/master/models/vgg.py
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import networks

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}



class VGG(networks.vgg.VGG):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__(cfg=cfg, batch_norm=batch_norm, num_classes=num_classes)




        self.bool_save_gradient=False
        
        target_layers=[]
        
        target_layers.append( len(self.block0._modules)-1)
        target_layers.append( len(self.block1._modules)-1)
        target_layers.append( len(self.block2._modules)-1)
        target_layers.append( len(self.block3._modules)-1)
        
        print(target_layers)
        self.target_layers=target_layers
        self.linear_attr = 'classifier'
        self.linear_input_dim= 512


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
            
        h = x.shape[2]
        x = F.relu(self.block0(x))

        if self.bool_save_gradient:
            x.register_hook(self.save_gradient) 
        conv_output_l1 = x

        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)

        if self.bool_save_gradient:
            x.register_hook(self.save_gradient)

        conv_output_l2 = x

        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)

        if self.bool_save_gradient:
            x.register_hook(self.save_gradient) 

        conv_output_l3 = x

        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)

        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        cam_feature = F.relu(x)
        x = self.pool4(cam_feature)
        feature = x.view(x.size(0), -1)
        out = self.classifier(feature)
        


        return out,feature,cam_feature,[conv_output_l1,conv_output_l2,conv_output_l3]



cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8_1028(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn_1028(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11_1028(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn_1028(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13_1028(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn_1028(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16_1028(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn_1028(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19_1028(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn_1028(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


# if __name__ == '__main__':
#     import torch

#     x = torch.randn(2, 3, 32, 32)
#     net = vgg19_bn(num_classes=100)
#     feats, logit = net(x, is_feat=True, preact=True)

#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)

#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')


'''
x = F.relu(self.block0(x))
torch.Size([6, 64, 32, 32])
x = self.pool0(x)
torch.Size([6, 64, 16, 16])
x = self.block1(x)
torch.Size([6, 128, 16, 16])
x = self.pool1(x)
torch.Size([6, 128, 8, 8])
x = self.block2(x)
torch.Size([6, 256, 8, 8])
x = self.pool2(x)
torch.Size([6, 256, 4, 4])
x = self.block3(x)
torch.Size([6, 512, 4, 4])
x = self.block4(x)
torch.Size([6, 512, 4, 4])
x = self.pool4(x)
torch.Size([6, 512, 1, 1])
features = x.view(x.size(0), -1)
torch.Size([6, 512])

'''

'''
x = F.relu(self.block0(x))
torch.Size([6, 64, 32, 32])
x = self.pool0(x)
torch.Size([6, 64, 16, 16])
x = self.block1(x)
torch.Size([6, 128, 16, 16])
x = self.pool1(x)
torch.Size([6, 128, 8, 8])
x = self.block2(x)
torch.Size([6, 256, 8, 8])
x = self.pool2(x)
torch.Size([6, 256, 4, 4])
x = self.block3(x)
torch.Size([6, 512, 4, 4])
x = self.block4(x)
torch.Size([6, 512, 4, 4])
x = self.pool4(x)
torch.Size([6, 512, 1, 1])
features = x.view(x.size(0), -1)
torch.Size([6, 512])


cam:  bs 10 4  4
gradcam:
torch.Size([6, 64, 32, 32])

torch.Size([6, 128, 16, 16])

torch.Size([6, 256, 8, 8])

torch.Size([6, 512, 4, 4])


ret_cams.size()
torch.Size([bs, 10, 4, 4])


showgradcams
torch.Size([1, 32, 32])
torch.Size([1, 16, 16])
torch.Size([1, 8, 8])
torch.Size([1, 4, 4])

'''