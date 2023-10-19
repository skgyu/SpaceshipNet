import os
import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from skyu_tools  import skyu_util as sutil
import csv
import time,copy
from  matplotlib import pyplot as plt
plt.switch_backend('agg')
import networks,os

class BaseModel(nn.Module):


    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.namedir = opt.expr_dir

        self.save_dir= self.namedir
        sutil.makedirs(self.save_dir)

        sutil.get_logger( os.path.join(opt.debug_info_dir, 'cls.log') , 'cls.log')


        plt.rcParams['savefig.dpi'] = 300  
        plt.rcParams['figure.dpi'] =  300 




    def save_network2(self, network, network_label, epoch_label, gpu_ids):

 

        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)


        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.to(self.opt.device)



    def delete_best(self):


        lst=os.listdir(self.save_dir)

        for file in lst:

            fileloc=os.path.join( self.save_dir, file)

            if  os.path.isfile(fileloc) and file.startswith('best') and file.endswith('.pth'):
                
                os.remove(fileloc)







    def save_network(self, network, network_label, epoch_label, gpu_ids):

        if self.opt.local_rank!=0:
            return

        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)


        torch.save(network.state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.to(self.opt.device)

            

    def load_network(self, network, network_label, epoch_label):

        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)



        print(save_path)
        if not os.path.exists(save_path):
            print('{} not load from previous'.format(network_label)  )
            print('{} not load from previous'.format(network_label)  )
            sutil.log('{} not load from previous'.format(network_label),'main')
            sutil.log('{} not load from previous'.format(network_label),'main')
            raise(RuntimeError('load error'))
            return False

        network.load_state_dict(torch.load(save_path))

        return True

    def save_scheduler(self, scheduler,scheduler_label, epoch_label):
        save_filename = '%s_scheduler_%s.pth' % (epoch_label, scheduler_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(scheduler.state_dict(), save_path)

    def save_optimizer(self, optimizer,optimizer_label, epoch_label):
        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)



    def load_optimizer(self, optimizer,optimizer_label, epoch_label):

        save_filename = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)

        print(save_path)
        if not os.path.exists(save_path):
            print('{} not load from previous'.format(optimizer_label)  )
            sutil.log('{} not load from previous'.format(optimizer_label),'main')
            print('{} not load from previous'.format(optimizer_label)  )
            sutil.log('{} not load from previous'.format(optimizer_label),'main')
            raise(RuntimeError('load error'))
            return False
        optimizer.load_state_dict(torch.load(save_path))

        return True    


 
    def load_optimizer_frompath(self, optimizer,load_path):

        if not os.path.exists(load_path):
            print('{} have not been found and not been loaded'.format(load_path)  )
            sutil.log('{} have not been found and not been loaded'.format(load_path),'main')
            raise(RuntimeError('load error'))
            return False
        optimizer.load_state_dict(torch.load(load_path))

        return True
  
   

    def load_network_frompath(self, network, load_path):

        if not os.path.exists(load_path):
            print('{} have not been found and not been loaded'.format(load_path)  )
            sutil.log('{} have not been found and not been loaded'.format(load_path),'main')
            raise(RuntimeError('load error'))
            return False

        network.load_state_dict(torch.load(load_path))

        return True



    def update_learning_rate(self , epoch):
        for scheduler in self.schedulers:
            scheduler.step()
        

        self.record_learning_rate(epoch)

    def record_learning_rate(self,epoch):

        

        loc=os.path.join(self.opt.expr_dir, 'learning_rate.txt')
        


        with open(loc, 'a') as f:

            f.write(  'epoch = {}\n'.format(epoch)  )

            for optimizer_name in self.optimizer_names:

                tmp_optimizer= getattr(self, optimizer_name)

                for param_id,param in enumerate(tmp_optimizer.param_groups):

                    lr = param['lr']

                    f.write( '{}'.format(optimizer_name) + 'param_id = {}'.format(param_id)   + ' , learning rate = %.7f\n' % lr)

                    print('lr={}'.format(lr))

        f.close()

        self.current_epoch=epoch




    def clear_loss(self):

        self.sum_loss= OrderedDict()
        self.cnt_loss= OrderedDict()
        self.accuracy_forG= 0
        self.sizes_C_forG = 0


        self.accuracy_forS= 0
        self.sizes_C_forS = 0
        


    def update_loss(self,key,add,addcnt=1):

        if key not  in  self.sum_loss:
            self.sum_loss[key]=0
            self.cnt_loss[key]=0

        self.sum_loss[key]+=  add
        self.cnt_loss[key]+=  addcnt


    def get_image_paths(self):
        return self.image_paths

    def record_ave_errors(self,epoch=None):

        if epoch is None and hasattr(self,"epoch"):
            epoch=self.epoch


        loc=os.path.join(self.opt.expr_dir, '{}_rank{}_loss_record.csv'.format(self.opt.name,self.opt.local_rank) )

        f=open(loc, 'a')
        writer = csv.writer(f)

        ret_item=['epoch']
        ret_value=[epoch if epoch is not None else '?']

        for key,value in self.sum_loss.items():
            ret_item.append(key)

            if self.cnt_loss[key]!=0:
                ret_value.append(self.sum_loss[key]/self.cnt_loss[key])
            else:
                ret_value.append(0)

                
        writer.writerow(ret_item)
        writer.writerow(ret_value)
        f.close()

        csvfile=open(loc,'r')
        reader=csv.reader(csvfile)

        loss_names=[]
        losses={}
        epoches={}


        for i,line in enumerate(reader):
            

            if i%2==0:

                for loss_name in line[1:]:
                    loss_names.append(loss_name)
                    if  loss_name not in losses.keys():
                        losses[loss_name]=[]
                        epoches[loss_name]=[]

                continue
            

            assert(len(loss_names)== len(line[1:]) )

            for loss_name in loss_names:
                epoches[loss_name].append( float(line[0]) )


            for i,loss_num in enumerate(line[1:]):
                loss_name= loss_names[i]
                losses[loss_name].append( float(loss_num) )
                

            loss_names=[]

        csvfile.close()
        


        plt.figure()
        plt.title("losses During Training")

        assert(losses.keys()==epoches.keys())

        keys=losses.keys()

        for loss_name in keys:
            plt.plot(epoches[loss_name],losses[loss_name],label=loss_name)


        plt.xlabel("trained epochs")
        plt.ylabel("losses")
        plt.legend()
        saveloc=os.path.join(self.opt.debug_info_dir, '{}_losses_{}epochs.png'.format(self.opt.name,epoch) )
        plt.savefig( saveloc) 
        plt.clf()
        



    def get_ave_errors(self,epoch=None):

        self.record_ave_errors(epoch)
        ret=copy.deepcopy(self.sum_loss)
        for key,value in ret.items():
             ret[key]/=self.cnt_loss[key]
        return  ret

    def save(self, epoch):

        self.saveall(epoch)


    def saveall(self, epoch):


        for name in self.principle_names+self.auxiliary_names:
            ret=getattr(self, name)
            self.save_network(ret,name, epoch , self.gpu_ids )

        for i,name in enumerate(self.optimizer_names):
            ret=getattr(self,name)
            self.save_optimizer(ret,name, epoch)        

            self.save_scheduler(self.schedulers[i],'scheduler_{}'.format(i), epoch)



    def load_scheduler(self, scheduler,scheduler_label, epoch_label):

        save_filename = '%s_scheduler_%s.pth' % (epoch_label, scheduler_label)
        save_path = os.path.join(self.save_dir, save_filename)

        print(save_path)
        if not os.path.exists(save_path):
            print('{} not load from previous'.format(scheduler_label)  )
            sutil.log('{} not load from previous'.format(scheduler_label),'main')
            print('{} not load from previous'.format(scheduler_label)  )
            sutil.log('{} not load from previous'.format(scheduler_label),'main')
            raise(RuntimeError('load error'))
            return False

        scheduler.load_state_dict(torch.load(save_path))

        return True


    def init_loss(self):

        self.criterion_L1=torch.nn.L1Loss()
        self.criterion_MSE=torch.nn.MSELoss()

        self.kl_loss =torch.nn.KLDivLoss()
        self.ce_loss=torch.nn.CrossEntropyLoss()

        self.LogSoftmax=nn.LogSoftmax(dim=1)
        self.NLLLoss=nn.NLLLoss()



    def init_scheduler(self):

        self.optimizers = []
        for name in self.optimizer_names:
            if hasattr(self,name):
                self.optimizers.append(getattr(self,name))

        self.schedulers = []
        for optimizer_name in self.optimizer_names:
            optimizer=getattr(self, optimizer_name)
            self.schedulers.append(networks.get_scheduler(optimizer, optimizer_name, self.opt))

