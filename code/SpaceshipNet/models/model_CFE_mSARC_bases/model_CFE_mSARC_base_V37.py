#-*- coding:utf-8 -*-
import torch
from collections import OrderedDict
from torch.autograd import Variable
from ..base_model import BaseModel
import networks
from PIL import Image
import copy,random,os,time,ntpath,ntpath,itertools
import numpy as np
import pandas as pd
import logging,sys
from torch.nn import DataParallel as DPL
from skyu_tools import skyu_util as sutil
import importlib
import glob
import  torchvision
from tqdm import tqdm
from skyu_tools import net_magnifier as smag
import torch.nn as nn
import functools
import torch.nn.functional as F
from torchvision import transforms
from kornia import augmentation
from datafree.utils import DataIter
from timm.data.mixup import Mixup



def gradcam_cope(self,net,conv_outputs,model_output,one_hot_labels,retain_graph):

    
    grad = torch.autograd.grad(model_output, net.parameters(), grad_outputs =one_hot_labels,retain_graph=retain_graph)
    
    cams=[]

    gradients= net.get_gradient()
    
    for gradient,conv_output in zip(gradients[::-1],conv_outputs):

        weights = gradient.mean([2,3]) # Take averages for each gradient
        cams.append( (weights.unsqueeze(2).unsqueeze(3)* conv_output).sum(dim=1) )  

    return cams

def makeup_fclikeconv(self,net,fc_like_conv):
    
    w_pa=None
    for name,pa in getattr(net, net.linear_attr).named_parameters():
        if name=='weight':
            w_pa=pa
            break

    assert(w_pa is not None)
            

    for x in fc_like_conv.parameters():
        x.data= w_pa.unsqueeze(2).unsqueeze(3)
        break
    
    

def cope_fclikeconv(self,net,cam_feature,fc_like_conv, makeup=True):
    

    if makeup:

        w_pa=None
        for name,pa in  getattr(net, net.linear_attr).named_parameters():
            if name=='weight':
                w_pa=pa
                break

        assert(w_pa is not None)
                

        for x in fc_like_conv.parameters():
            x.data= w_pa.unsqueeze(2).unsqueeze(3)
            break
    
    
    ret_cams= fc_like_conv(cam_feature)
    
    return ret_cams



def initializeB(self,opt):



    sutil.log('model_CFE_mSARC_base_V37','main')
    print('model_CFE_mSARC_base_V37')


    ##################################################################################
    self.principle_names=['student_net']
    self.auxiliary_names=['generator','teacher_net']
    self.hiddens=[]
    self.optimizer_names=['optimizer_S']
    ##################################################################################




    self.student_net= networks.find_network(opt['student_net'])

    if opt.isTrain:
        self.teacher_net =  networks.find_network(opt['teacher_net'])
        self.teacher_net.eval()
        self.generator =     networks.find_network(opt['generator'])


    #################
    if not self.isTrain or opt.continue_train:  ##### test or continue training
        which_epoch = opt.which_epoch 
        self.load_network(self.student_net, 'student_net', which_epoch)
    else:  #####   train from scratch  
        print('#########train from scrach#############')
        if 'init' in opt['student_net'] and opt['student_net']['init'] is not None:
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
        self.kl_loss_criterion = torch.nn.KLDivLoss(reduction='batchmean') 



        self.init_optS()

        
        if opt.continue_train:  
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

        self.fc_like_conv_teacher_tmp = nn.Conv2d(self.teacher_net.linear_input_dim, opt.teacher_net.num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(opt.device)
        self.fc_like_conv_teacher_tmp.eval()    

        self.fc_like_conv_student_tmp = nn.Conv2d(self.student_net.linear_input_dim, opt.teacher_net.num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(opt.device)
        self.fc_like_conv_student_tmp.eval()    



        self.fc_like_conv_teacher = nn.Conv2d(self.teacher_net.linear_input_dim, opt.teacher_net.num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(opt.device)
        self.fc_like_conv_teacher.eval()
    

        self.fc_like_conv_student = nn.Conv2d(self.student_net.linear_input_dim, opt.teacher_net.num_classes, kernel_size=1, stride=1,padding=0, bias=False).to(opt.device)
        self.fc_like_conv_student.eval()


        self.loss_meanvar_feature_layers = []

        for module in self.teacher_net.modules():

            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
                self.loss_meanvar_feature_layers.append(smag.DeepInversionFeatureHook(module))




        self.aug =    transforms.Compose([ 
                augmentation.RandomCrop(size=[opt.generator.img_size, opt.generator.img_size], padding=opt.aug_padding),
                augmentation.RandomHorizontalFlip()
            ])


        self.dataset_transform=  transforms.Compose([
            transforms.RandomCrop(opt.generator.img_size, padding=opt.aug_padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])


        self.gdataset_dir = os.path.join( opt.expr_dir,'synthesis')
        sutil.makedirs(self.gdataset_dir)

        self.glmdb_dir = os.path.join( opt.expr_dir,'feature')
        sutil.makedirs(self.glmdb_dir)


        modellib = importlib.import_module('datafree.utils._utils')

        ImagePoolFunc= None     
        for name, cls in modellib.__dict__.items():
            if name  == opt.FeatureLabelImagePoolType:
                ImagePoolFunc = cls



        if ImagePoolFunc is  None:     
            modellib = importlib.import_module('datafree.utils._utils_skyu')
            for name, cls in modellib.__dict__.items():
                if name  == opt.FeatureLabelImagePoolType:
                    ImagePoolFunc = cls


        self.data_pool0 = ImagePoolFunc(root=self.glmdb_dir, name='feature_label_images0.lmdb')
        self.data_pool1 = ImagePoolFunc(root=self.glmdb_dir, name='feature_label_images1.lmdb')
        self.data_pool2 = ImagePoolFunc(root=self.glmdb_dir, name='feature_label_images2.lmdb')

        
        if opt.use_mixup_cutmix:
            self.mixup_fn= Mixup(**opt.mixup_args)

    

def record_feature_fuse(self,
    mix_syn_images,
    syn_images,
    labels,
    images,
    epoch,
    conv_idx,
    pre='',
    save_per_image=False,
    teacher_out_images=None,
    teacher_out_syn_images=None,
    teacher_out_mix_syn_images=None):
    

    record_teacher_label=False
    if teacher_out_images is not None:
        assert(teacher_out_syn_images is not None)
        assert(teacher_out_mix_syn_images is not None)
        record_teacher_label=True



    epoch=str(epoch)


    debug_image_dir= os.path.join(self.opt.debug_image_dir,epoch)
    sutil.makedirs(debug_image_dir)


    typename=    pre 
    sutil.log('type='+typename,'sample')


    opt=self.opt

    labels=labels.cpu().numpy()

    images=images.detach()

    showimages=[]

    assert(images.size()[0]==mix_syn_images.size()[0])
    assert(syn_images.size()[0]==mix_syn_images.size()[0])


    nimg=images.size()[0]

    H= images.size()[2]

    for i in range(nimg):

        showimages.append(images[i])
        showimages.append(images[nimg-1-i])
        showimages.append(syn_images[i])
        showimages.append(syn_images[nimg-1-i])
        showimages.append(mix_syn_images[i])


    tensor_image=torch.stack(showimages,dim=0)

    n_row=5

    image_grid=torchvision.utils.make_grid(tensor_image, nrow=n_row, padding=0, normalize=True, range=None, scale_each=True, pad_value=0)
    image_grid=image_grid*2-1


    image_numpy=sutil.tensor2im(image_grid)
    sutil.save_image( image_numpy,  os.path.join(   debug_image_dir  ,'grid_a_b_sa_sb_mixfeasynimage_eachnorm.png'   ) )
    sutil.save_image( image_numpy[:10*H],  os.path.join(   debug_image_dir  ,'grid_a_b_sa_sb_mixfeasynimage_eachnorm_0-9.png'   ) )


    image_grid=torchvision.utils.make_grid(tensor_image, nrow=n_row, padding=0, normalize=False, range=None, scale_each=False, pad_value=0)
    image_grid=image_grid*2-1
    image_numpy=sutil.tensor2im(image_grid)
    sutil.save_image( image_numpy,  os.path.join(   debug_image_dir  ,'grid_a_b_sa_sb_mixfeasynimage_nonorm.png'   ) )
    sutil.save_image( image_numpy[:10*H],  os.path.join(   debug_image_dir  ,'grid_a_b_sa_sb_mixfeasynimage_nonorm_0-9.png'   ) )


    txtloc = os.path.join(   debug_image_dir  ,'a_b_sa_sb_mixfeasynimage_info.txt'.format(typename)   )


    with open(txtloc,'w') as fw:

        opt=self.opt


        fw.write('imagea imageb synimagea synimageb mix_syn_image\n' )
        fw.write('conv_idx = %d\n'% conv_idx )


        for st in range(0, nimg ,n_row):
            ed=st+n_row

            ed=min(ed,nimg)
            for i in range(st,ed):
                fw.write('idx {} a: {} b: {}\n'.format(i, labels[i], labels[nimg-1-i]) )
            fw.write('\n')




        fw.write('cated tensor_image\n')        
        fw.write('min max:\n')        
        fw.write('{} {}\n'.format( torch.min(tensor_image).item(), torch.max(tensor_image).item()  ) )

        fw.write('image_grid after*2 -1(normalize=True, range=None, scale_each=True)\n')        
        fw.write('min max:\n')        
        fw.write('{} {}\n'.format( torch.min(image_grid).item(), torch.max(image_grid).item()  ) )
   

    #####

    fw.close()



    teacher_out_images_softmax=torch.nn.functional.softmax(teacher_out_images)
    teacher_out_syn_images_softmax=torch.nn.functional.softmax(teacher_out_syn_images)
    teacher_out_mix_syn_images_softmax=torch.nn.functional.softmax(teacher_out_mix_syn_images)

    if record_teacher_label:
        loggername=  '{}_a_b_sa_sb_mixfeasynimage_teacher_out'.format(epoch)
        sutil.get_logger(path=os.path.join(debug_image_dir, 'a_b_sa_sb_mixfeasynimage_teacher_out.log' ),name=loggername )        

        softmaxloggername=  '{}_a_b_sa_sb_mixfeasynimage_teacher_out_softmax'.format(epoch)
        sutil.get_logger(path=os.path.join(debug_image_dir, 'a_b_sa_sb_mixfeasynimage_teacher_out_softmax.log' ),name=softmaxloggername )

        for ia in range(nimg):

            ib= nimg-1-ia

            sutil.log(  'idx={}'.format(ia) ,loggername)
            sutil.log(  'idx={}'.format(ia) ,softmaxloggername)

            sutil.log(  'teacher out image a=',loggername)
            sutil.log(  teacher_out_images[ia],loggername)         
            sutil.log(  'teacher out softmax image a=',softmaxloggername)
            sutil.log(  teacher_out_images_softmax[ia],softmaxloggername)

            sutil.log(  'teacher out image b=',loggername)
            sutil.log(  teacher_out_images[ib],loggername)          
            sutil.log(  'teacher out softmax image b=',softmaxloggername)
            sutil.log(  teacher_out_images_softmax[ib],softmaxloggername)

            sutil.log(  'teacher out syn_image a=',loggername)
            sutil.log(  teacher_out_syn_images[ia],loggername)            
            sutil.log(  'teacher out softmax syn_image a=',softmaxloggername)
            sutil.log(  teacher_out_syn_images_softmax[ia],softmaxloggername)

            sutil.log(  'teacher out syn_image b=',loggername)
            sutil.log(  teacher_out_syn_images[ib],loggername)            
            sutil.log(  'teacher out softmax syn_image b=',softmaxloggername)
            sutil.log(  teacher_out_syn_images_softmax[ib],softmaxloggername)

            sutil.log(  'teacher out mix_feature_syn_image=',loggername)
            sutil.log(  teacher_out_mix_syn_images[ia],loggername)            
            sutil.log(  'teacher out softmax mix_feature_syn_image=',softmaxloggername)
            sutil.log(  teacher_out_mix_syn_images_softmax[ia],softmaxloggername)



    torchvision.utils.save_image(tensor_image.clone(),
                        os.path.join(   debug_image_dir  ,'usave_a_b_sa_sb_mixfeasynimage_eachnorm.png'   ) ,
                        normalize=True, scale_each=True, nrow=n_row)    

    torchvision.utils.save_image(tensor_image[:10*n_row].clone(),
                        os.path.join(   debug_image_dir  ,'usave_a_b_sa_sb_mixfeasynimage_eachnorm_0-9.png'  ) ,
                        normalize=True, scale_each=True, nrow=n_row)

    torchvision.utils.save_image(tensor_image.clone(),
                        os.path.join(   debug_image_dir  ,'usave_a_b_sa_sb_mixfeasynimage_nonorm.png'   ) ,
                        normalize=False, scale_each=False, nrow=n_row)    

    torchvision.utils.save_image(tensor_image[:10*n_row].clone(),
                        os.path.join(   debug_image_dir  ,'usave_a_b_sa_sb_mixfeasynimage_nonorm_0-9.png'   ) ,
                        normalize=False, scale_each=False, nrow=n_row)

    if save_per_image:


        savedir=os.path.join(   debug_image_dir, pre+'_imgs')
        sutil.makedirs( savedir  )

        for i in range(5):

            tmp_img=images[i].clone().detach()
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)
            saveloc=os.path.join(   savedir ,'imagea_%02d.png'%i   )
            sutil.save_image( image_numpy,  saveloc )


            tmp_img=images[nimg-1-i].clone().detach()
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)
            saveloc=os.path.join(   savedir ,'imageb_%02d.png'%i   )
            sutil.save_image( image_numpy,  saveloc )



            tmp_img=syn_images[i].clone().detach()
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)
            saveloc=os.path.join(   savedir ,'synimagea_%02d.png'%i   )
            sutil.save_image( image_numpy,  saveloc )

            tmp_img=syn_images[nimg-1-i].clone().detach()
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)
            saveloc=os.path.join(   savedir ,'synimageb_%02d.png'%i   )
            sutil.save_image( image_numpy,  saveloc )         


            tmp_img=mix_syn_images[i].clone().detach()
            tmp_img=tmp_img*2-1
            image_numpy=sutil.tensor2im(tmp_img)
            saveloc=os.path.join(   savedir ,'mix_syn_image_%02d.png'%i   )
            sutil.save_image( image_numpy,  saveloc )


def train_G(self):

    opt=self.opt


    num_classes=opt.teacher_net.num_classes


    best_cost = 1e6
    best_inputs = None

    z = torch.randn(size=(opt.batch_size, opt.generator.input_dim), device=opt.device).requires_grad_() 



    self.generate_input(test=True,labels= None)


    for mod in self.loss_meanvar_feature_layers:
        mod.calloss(True)


    optimizer = torch.optim.Adam([ {'params': self.generator.parameters()} ], lr=opt['optimizer_G']['lr'][0], betas=[0.5, 0.999] ) 

    batch_size=opt.batch_size
    num_classes=opt.teacher_net.num_classes


    self.makeup_fclikeconv(self.teacher_net,self.fc_like_conv_teacher)
    self.makeup_fclikeconv(self.student_net,self.fc_like_conv_student)

    for it in range(opt.g_steps):

        verbose=False
        if  it%20==0:
            print('function [synthesize] it/opt.g_steps:  %d/%d'%(it,opt.g_steps) )
            verbose=True


        fake_data, conv_outs=self.generator(z)


        correct_cnt=-1



        inputs_jit = self.aug(fake_data)

        normalized_inputs_jit=self.normalize_before_cls(inputs_jit)  ##### 

        teacher_out,teacher_feature,teacher_cam_feature,teacher_conv_output = self.teacher_net(normalized_inputs_jit)



        cls_loss=   self.ce_loss( teacher_out, self.labels)* opt.lambda_cls

        self.sizes_C_forG+= opt.batch_size


        correct_cnt=(teacher_out.max(1)[1]).eq(self.labels).sum().item()

        self.accuracy_forG+=     correct_cnt

        

        if it == opt.g_steps - 1:

            self.correct_forG_last_iter= correct_cnt
            self.batch_size_forG_last_iter= opt.batch_size



        if verbose:
            print('classify_loss_forG idx correct num= {}/{} '.format( correct_cnt, opt.batch_size) )


        self.update_loss(  'cls_loss', cls_loss.item() )
        if verbose:
            print('cls_loss {}'.format(cls_loss.item() ) ) 


        loss_bn_meanvar=0
        if opt.lambda_bn_meanvar>0:

            rescale = [opt.first_bn_multiplier] + [1. for _ in range(len(self.loss_meanvar_feature_layers)-1)]
            loss_bn_meanvar = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_meanvar_feature_layers)]) * opt.lambda_bn_meanvar
            

            self.update_loss(  'loss_bn_meanvar', loss_bn_meanvar.item() )
            if verbose:
                print('loss_bn_meanvar {}'.format(loss_bn_meanvar.item() ) ) 



        loss_G =    cls_loss  + loss_bn_meanvar 
                
        


        with torch.no_grad():
            if  best_inputs is None or best_cost > loss_G.item():
                best_cost = loss_G.item()

                layer1_out,conv1_out,conv2_out = conv_outs

                best_inputs = fake_data
                best_conv_out  = [layer1_out.clone().detach(), conv1_out.clone().detach(), conv2_out.clone().detach()]

        optimizer.zero_grad()
        loss_G.backward()
        optimizer.step()

    


    print(best_conv_out[0].cpu().numpy().shape)
    print(best_conv_out[1].cpu().numpy().shape)
    print(best_conv_out[2].cpu().numpy().shape)
    print( self.labels.cpu().numpy().shape)


    self.data_pool0.add( best_conv_out[0].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )
    self.data_pool1.add( best_conv_out[1].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )
    self.data_pool2.add( best_conv_out[2].cpu().numpy(), self.labels.cpu().numpy(), best_inputs.clone().detach().clamp(0, 1).cpu().numpy() )



    dst0 = self.data_pool0.get_dataset()
    dst1 = self.data_pool1.get_dataset()
    dst2 = self.data_pool2.get_dataset()


    train_sampler=None
    loader0 = torch.utils.data.DataLoader(
        dst0, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)    

    loader1 = torch.utils.data.DataLoader(
        dst1, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    loader2 = torch.utils.data.DataLoader(
        dst2, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    self.data_iter0 = DataIter(loader0)
    self.data_iter1 = DataIter(loader1)
    self.data_iter2 = DataIter(loader2)

    self.fake_data=  best_inputs.clone().detach()




def channel_mix_feature(self, a, b):


    nc=a.size()[0]
    picks=np.random.choice(nc, nc//2, replace=False)
    se2=set(range(nc))-set(picks)
    unpicks=list(se2)
    cmask1=torch.zeros([nc], device=self.opt.device).scatter_(0, torch.LongTensor(picks).cuda(), 1)
    cmask2=torch.zeros([nc], device=self.opt.device).scatter_(0, torch.LongTensor(unpicks).cuda(), 1)
    viewidxs= [nc]+[1 for i in range( len(list(a.size()))-1  ) ]
    aug_feature=a*cmask1.view(*viewidxs)+  b*cmask2.view(*viewidxs)  

    return aug_feature


def  channel_mix_batch(self,feature,labels):


    nfeature=feature.size()[0]

    ansfeature= feature.clone().detach()
    anslabels= labels.clone().detach()

    for i in range(nfeature):

        j=nfeature-1-i

        ansfeature[i]=self.channel_mix_feature(feature[i],feature[j])

        anslabels[i]= (labels[i]+labels[j])/2

    return ansfeature,anslabels



def train_S(self):


    opt=self.opt

    num_classes=opt.teacher_net.num_classes

    for mod in self.loss_meanvar_feature_layers:
        mod.calloss(False)


    for it in range(opt.kd_steps):

        verbose=False
      
        conv_idx =  int(np.random.choice(opt.conv_idx, 1)[0])

        if conv_idx==0:
            tmp_data = self.data_iter0.next()

        elif conv_idx==1:
            tmp_data = self.data_iter1.next()

        elif conv_idx==2:
            tmp_data = self.data_iter2.next()

        else:
            raise(RuntimeError('no such conv_idx'))

        if it%20==0:
            print('function [train] it/opt.kd_steps:  %d/%d'%(it,opt.kd_steps) )
            print('conv_idx = %d'% conv_idx )
            verbose=True

    
        features= tmp_data['feature']
        labels= tmp_data['label']
        images= tmp_data['image']

        features=features.to(opt.device)
        labels=labels.to(opt.device)
        images=images.to(opt.device)


        with torch.no_grad():

            conv_out_mix = features
            if opt.use_gen_mixup_cutmix and np.random.rand()<opt.gen_mixup_cutmix_p:

                if opt.aug_typ=='channel_mix':
                    conv_out_mix, mixlabels = self.channel_mix_batch( features   , labels ) 
                else:
                    raise(RuntimeError('aug_typ is not known'))


                mix_syn_images  = self.generator.generate_using_convout(conv_out_mix, conv_idx)



                
                if not self.save_cutmix_mixup:

                    if opt.aug_typ=='channel_mix_important':
        
                        nc=features.size()[1]
                        nseg=opt['aug_par']['nseg']  
                        assert(nc%nseg==0)



                    syn_images  =self.generator.generate_using_convout(features, conv_idx)


                    teacher_out_images        ,__,__,__ = self.teacher_net(self.normalize_before_cls(images) )
                    teacher_out_syn_images    ,__,__,__ = self.teacher_net(self.normalize_before_cls(syn_images) )
                    teacher_out_mix_syn_images,__,__,__ = self.teacher_net(self.normalize_before_cls(mix_syn_images) )

                    self.record_batchimg(mix_syn_images.clone().detach(), self.iepoch+1, pre='cutmix_mixup_result_images_epoch{}'.format(self.iepoch+1)  )


                    self.record_feature_fuse(mix_syn_images=mix_syn_images.clone().detach(),
                        syn_images=syn_images.clone().detach(),
                        labels=labels.clone().detach(),
                        images=images.clone().detach(),
                        epoch=self.iepoch+1,
                        conv_idx=conv_idx,
                        pre='ana_cutmix_mixup_result_images_epoch{}'.format(self.iepoch+1),
                        save_per_image=False,
                        teacher_out_images=teacher_out_images,
                        teacher_out_syn_images=teacher_out_syn_images,
                        teacher_out_mix_syn_images=teacher_out_mix_syn_images  )
                    self.save_cutmix_mixup=True


            else:
                mix_syn_images  = self.generator.generate_using_convout(conv_out_mix, conv_idx)



            mix_syn_images = self.aug(mix_syn_images)


        suffix='generate'

        normalized_fake_data=self.normalize_before_cls(mix_syn_images)


        if opt.use_mixup_cutmix and np.random.rand()<opt.mixup_cutmix_p:

            with torch.no_grad():
                teacher_out,teacher_feature,teacher_cam_feature,teacher_conv_output = self.teacher_net(normalized_fake_data)
                teacher_labels=teacher_out.max(1)[1]
                inputs_mix, classes_mix = self.mixup_fn( images.clone().detach()   , teacher_labels.clone().detach() )
                fake_data=inputs_mix.clone()
                normalized_fake_data=self.normalize_before_cls(fake_data)



        self.teacher_net.do_save_gradient()
        self.student_net.do_save_gradient()

        
        teacher_out,teacher_feature,teacher_cam_feature,teacher_conv_output = self.teacher_net(normalized_fake_data)

        student_out,student_feature,student_cam_feature,student_conv_output = self.student_net(normalized_fake_data)

        kl_loss=0

        T=opt.distill.T


        if 'alpha_2' in opt.distill:
            kl_lambda = opt.distill.alpha_2
        else:
            alpha= opt.distill.alpha
            kl_lambda = alpha * T * T


        if kl_lambda>0:

            kl_loss=nn.KLDivLoss()(F.log_softmax(student_out/T, dim=1),
                                 F.softmax(teacher_out/T, dim=1)) * kl_lambda

            self.update_loss(  'kl_loss_{}'.format(suffix), kl_loss.item() )

            if verbose:
                print('kl_loss_{} {}'.format(suffix,kl_loss.item() ) ) 


        loss_mSARC_cam_mse=0

        if opt.lambda_mSARC_cam_mse>0:

            tea_fc_like_conv_cams=self.cope_fclikeconv(self.teacher_net,teacher_cam_feature, self.fc_like_conv_teacher_tmp)
            stu_fc_like_conv_cams=self.cope_fclikeconv(self.student_net,student_cam_feature, self.fc_like_conv_student_tmp)

            loss_mSARC_cam_mse=self.criterion_MSE(stu_fc_like_conv_cams,tea_fc_like_conv_cams) *opt.lambda_mSARC_cam_mse


            self.update_loss(  'loss_mSARC_cam_mse_{}'.format(suffix), loss_mSARC_cam_mse.item() )

            if verbose:
                print('loss_mSARC_cam_mse_{} {}'.format(suffix,loss_mSARC_cam_mse.item() ) ) 


        loss_mSARC_gradcams_mse=0
        if opt.lambda_mSARC_gradcams_mse>0:

            with torch.no_grad():
                one_hot_labels=torch.nn.functional.one_hot(teacher_out.max(1)[1],num_classes=num_classes)

            tea_cams=self.gradcam_cope(net=self.teacher_net,conv_outputs=teacher_conv_output,model_output=teacher_out,one_hot_labels=one_hot_labels,retain_graph=True)
            stu_cams=self.gradcam_cope(net=self.student_net,conv_outputs=student_conv_output,model_output=student_out,one_hot_labels=one_hot_labels,retain_graph=True)


            for stu_cam,tea_cam in zip(stu_cams,tea_cams):
                loss_mSARC_gradcams_mse+= self.criterion_MSE(stu_cam,tea_cam) * opt.lambda_mSARC_gradcams_mse

            self.update_loss(  'loss_mSARC_gradcams_mse_{}'.format(suffix), loss_mSARC_gradcams_mse.item() ) 

            if verbose:
                print('loss_mSARC_gradcams_mse_{} {}'.format(suffix,loss_mSARC_gradcams_mse.item() ) ) 
        

        loss=  kl_loss  + loss_mSARC_cam_mse + loss_mSARC_gradcams_mse

        self.optimizer_S.zero_grad()
        loss.backward()

        if opt.use_clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_net.parameters(), opt.lambda_clip_grad)

        self.optimizer_S.step()

        self.teacher_net.do_save_gradient(False)
        self.student_net.do_save_gradient(False)


def optimize_step(self):


    assert(self.student_net.training)

    self.save_cutmix_mixup=False



    self.generator.train()

    if self.iepoch%self.opt.synthesize_interval==0:
        self.train_G()


    self.generator.eval()

    self.train_S()


 