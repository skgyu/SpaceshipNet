from .init import *
from .blocks import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)



class CMIGeneratorV3(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(CMIGeneratorV3, self).__init__()


        self.layer_names=['l1','layer1','up_layer1','conv_layer1','up_layer2','conv_layer2','conv_layer3']


        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))
        
        self.layer1 =  nn.BatchNorm2d(ngf * 2)

        self.up_layer1 = nn.Upsample(scale_factor=2)
        

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True))


        self.up_layer2 = nn.Upsample(scale_factor=2)
        

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        layer1_out = self.layer1(out)
        out = self.up_layer1(layer1_out)
        conv1_out = self.conv_layer1(out)
        out = self.up_layer2(conv1_out)
        conv2_out = self.conv_layer2(out)
        img = self.conv_layer3(conv2_out)
        return img, [layer1_out,conv1_out,conv2_out]

    def generate_using_convout(self,feature_input,conv_idx):


        if conv_idx==0:

            out = self.up_layer1(feature_input)
            conv1_out = self.conv_layer1(out)
            out = self.up_layer2(conv1_out)
            conv2_out = self.conv_layer2(out)
            img = self.conv_layer3(conv2_out)


        elif conv_idx==1:
            out = self.up_layer2(feature_input)
            conv2_out = self.conv_layer2(out)
            img = self.conv_layer3(conv2_out)

        elif conv_idx==2:

            img = self.conv_layer3(feature_input)

        return img

    def reset_model(self,conv_idx):


        # assert (conv_idx in [0,1,2])


        if conv_idx==0:

            models = [self.l1, self.layer1]   

        elif conv_idx==1:

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1]

        elif conv_idx==2:

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1, self.up_layer2, self.conv_layer2]

        elif conv_idx=='all':

            models = [self.l1, self.layer1, self.up_layer1, self.conv_layer1, self.up_layer2, self.conv_layer2, self.conv_layer3]

        elif conv_idx=='none':

            models = []


        for model in models:        

            for m in model.modules():
                if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, (nn.BatchNorm2d)) or isinstance(m, nn.SyncBatchNorm):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)

                    





