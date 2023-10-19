#-*- coding:utf-8 -*-
import torch
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import argparse
import yaml
import shutil
from data.cifar10Dataset  import Cifar10Dataset
from data.cifar100Dataset  import Cifar100Dataset
from data.tinyimagenet_skyu  import SkyuTinyImageNet
from skyu_tools import skyu_util as sutil 
from options.train_options import TrainOptions
from models.models import create_model
from data.imagenette2Dataset import Imagenette2Dataset
from data.imagenet100Dataset import Imagenet100Dataset
import json

opt=TrainOptions().parse()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

print('####################')
print(str(opt.device))
print('####################')


if opt.dataset=='cifar-10':
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
elif opt.dataset=='cifar-100':
    from setting.class_names import cifar100_classnames
    classes=  cifar100_classnames['fine_label_names']
    coarse_classes=  cifar100_classnames['coarse_label_names']
elif opt.dataset=='tiny_imagenet':
    from setting.class_names import tiny_imagenet_classnames
    classes=  tiny_imagenet_classnames

elif opt.dataset=='PACS':
    from setting.class_names import PACS_classnames
    classes=  PACS_classnames

elif opt.dataset=='imagenette2':
    from setting.class_names import imagenette2_classnames
    classes=  imagenette2_classnames

elif opt.dataset=='imagenet100':
    with open(os.path.join(opt.dataroot,'imagenet100/Labels.json'), 'r', encoding='utf-8') as fw:
        classdict = json.load(fw)
        class_numbers=list(classdict.keys())
        classes=list(classdict.values())
        


def test(trainer,epoch,testloader,max_acc,test_net_name='student_net', best_epoch=-1):


    is_best_epoch=False

    trainer.evalstate()
    trainer.testC(epoch)


    print('trainer.student_net.training={}'.format(trainer.student_net.training))

    rs_txt= os.path.join( opt.expr_dir, 'results', 'local_rank='+str(opt.local_rank) ,'{}_epoch{}_result.txt'.format(test_net_name,epoch))
    rs_txt_all= os.path.join( opt.expr_dir, 'results','local_rank='+str(opt.local_rank),'{}_result.txt'.format(test_net_name))
    sutil.makedirs(  os.path.dirname(rs_txt) )

    correct = 0
    total = 0


    test_num=0
    with torch.no_grad():
        for data in testloader:
            test_num+=data['image'].size(0)
            trainer.set_input(data)
            outputs = trainer.predict(test_net_name=test_net_name)

            _, predicted = torch.max(outputs.data, 1)
            total += trainer.labels.size(0)
            correct += (predicted == trainer.labels).sum().item()

    print(test_num)


    tmp_acc=100 * correct / total

    if tmp_acc>max_acc:
        max_acc=tmp_acc
        best_epoch=epoch
        is_best_epoch=True


    print('Accuracy of the network on the test images after training %d epochs: %.2f %%\n' % (epoch ,
        100 * correct / total))

    with open(rs_txt,'a') as fw:
        fw.write('Accuracy of the network on the test images after training %d epochs: %.2f %%\n' % ( epoch ,100 * correct / total) )
    fw.close()    

    with open(rs_txt_all,'a') as fw:
        fw.write('Accuracy of the network on the test images after training %d epochs: %.2f %%\n' % ( epoch, 100 * correct / total) )
    fw.close()


    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}


    nclass=  len(classes)
    confused_matrix= np.zeros((nclass,nclass)).astype(np.int64)

    with torch.no_grad():
        for data in testloader:
            

            trainer.set_input(data)
            outputs = trainer.predict(test_net_name=test_net_name)

            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(trainer.labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

                confused_matrix[label][prediction]+=1


    for i,classname in enumerate(classes):

        assert(confused_matrix[i].sum() == total_pred[classname]  )
        assert(confused_matrix[i][i] == correct_pred[classname] )




    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))


        with open(rs_txt,'a') as fw:
            fw.write("Accuracy for class {:5s} is: {:.1f} %\n".format(classname, accuracy))
        fw.close()        

        with open(rs_txt_all,'a') as fw:
            fw.write("Accuracy for class {:5s} is: {:.1f} %\n".format(classname, accuracy))
        fw.close()


    print('max_acc = {:.2f}%'.format(max_acc) )

    with open(rs_txt,'a') as fw:
        fw.write('max_acc = {:.2f}%  best_epoch = {:5d}\n'.format(max_acc, best_epoch) )
        fw.write(' '*6)
        for i,classname in enumerate(classes):
            fw.write('%5s '%classname  )
        fw.write('\n')

        for i,classname in enumerate(classes):
            fw.write('%5s '%classname  )
            for j in range(nclass):
                fw.write('%5d '%confused_matrix[i][j]  )
            fw.write('\n')
    fw.close()    


    with open(rs_txt_all,'a') as fw:
        fw.write('max_acc = {:.2f}%  best_epoch = {:5d}\n'.format(max_acc, best_epoch) )
        fw.write(' '*6)
        for i,classname in enumerate(classes):
            fw.write('%5s '%classname  )
        fw.write('\n')

        for i,classname in enumerate(classes):
            fw.write('%5s '%classname  )
            for j in range(nclass):
                fw.write('%5d '%confused_matrix[i][j]  )
            fw.write('\n')
    fw.close()


    rs_csv= os.path.join( opt.expr_dir, 'results', 'local_rank='+str(opt.local_rank), 'confused_matrix_{}_epoch{}.csv'.format(test_net_name,epoch))

    import csv

    with open(rs_csv, 'w') as f:     
        writer = csv.writer(f)

        items=['']
        for i,classname in enumerate(classes):
            items.append(classname)
        writer.writerow(items)

        for i,classname in enumerate(classes):
            items=[classname]
            for j in range(nclass):
                items.append( confused_matrix[i][j] )
            items.append('%.2f%%'%(100.0*confused_matrix[i][i]/confused_matrix[i].sum() ) )
            writer.writerow(items)

    f.close()


    return max_acc,best_epoch,is_best_epoch



if __name__ =='__main__':


    
    if opt.dataset=='cifar-10':
        testset =  Cifar10Dataset(root=opt.dataroot, train=False, aug=False,normalize_transfer=opt.prin_normalize_func)
    elif opt.dataset=='cifar-100':
        testset =  Cifar100Dataset(root=opt.dataroot, train=False, aug=False,normalize_transfer=opt.prin_normalize_func)
    elif opt.dataset=='tiny_imagenet':    
        testset =  SkyuTinyImageNet(data_root=opt.dataroot, split='val')
    elif opt.dataset=='imagenette2':
        testset =  Imagenette2Dataset(root=opt.dataroot, phase='val', img_size=224, use_resize = True )

    elif opt.dataset=='imagenet100':
        testset =  Imagenet100Dataset(root=opt.dataroot, phase='val', img_size=224, use_resize = True )


    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=opt.num_workers, pin_memory=True)


    trainer=  create_model(opt)

    trainer.record_learning_rate(opt.start_epoch)

    max_acc=0
    best_epoch=-1

    
    iter_num=0

    
    if opt.continue_train:

        num_classes=opt['student_net']['num_classes']

        for epoch in range(0,opt.which_epoch+1):

            rs_txt= os.path.join( opt.expr_dir, 'results', 'local_rank='+str(opt.local_rank) ,'student_net_epoch{}_result.txt'.format(epoch) )

            with open(rs_txt,'r') as fr:
                for i in range(num_classes+2):
                    line = fr.readline()

                assert( line.strip().split(' ')[0]  =='max_acc'),line

                assert( line.strip().split(' ')[0]  =='max_acc'),line
                tmp_acc=float(line.strip().split(' ')[2][:-1])
            fr.close()

            if tmp_acc> max_acc:
                max_acc=tmp_acc
                best_epoch=epoch 

            
    print('max_acc={}'.format(max_acc))
    print('max_acc={}'.format(max_acc))
    print('max_acc={}'.format(max_acc))

    

    trainer.evalstate()

    if opt.start_epoch==0:
        test(trainer=trainer,epoch=opt.start_epoch,testloader=testloader,max_acc=max_acc,best_epoch=best_epoch,test_net_name='teacher_net')
        max_acc,best_epoch, is_best_epoch =test(trainer=trainer,epoch=opt.start_epoch,testloader=testloader,max_acc=max_acc,best_epoch=best_epoch)


    for epoch in range( opt.start_epoch,opt.max_epoch+1):  # loop over the dataset multiple times


        print(epoch)
        trainer.iepoch=epoch

        trainer.trainstate()
        print('trainer.student_net.training={}'.format(trainer.student_net.training))
        trainer.clear_loss()
        
        trainer.optimize_step( )

        
        if (epoch+1)%opt.synthesize_interval==0:
            trainer.record_batchimg(trainer.fake_data,epoch+1,pre='syn_fake_data_epoch{}'.format(epoch+1))



        trainer.evalstate()
        trainer.record_ave_errors(epoch= epoch  )



        if (epoch+1)%opt.save_epoch_freq ==0:
            trainer.saveall(epoch+1)
            
        max_acc,best_epoch, is_best_epoch=test(trainer=trainer,epoch=epoch+1,testloader=testloader,max_acc=max_acc,best_epoch=best_epoch)

        if is_best_epoch:
            trainer.delete_best()
            trainer.saveall('best'+str(epoch+1))

        trainer.update_learning_rate(epoch+1)


    print('Finished Training')








  
  



