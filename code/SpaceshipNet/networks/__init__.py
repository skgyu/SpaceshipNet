from .init import *
from .scheduler import *
import importlib
from functools import partial
from .resnet_cifar import *


def find_network(network_args):


    typ=network_args['arc']

    args={}


    if typ in ['CMIGenerator224All']:
        network_py= find_file_using_name(dirs='networks', model_name='cmi_generator_DG')  
        classname=typ
        if 'input_dim' in network_args:
            args['nz']=network_args['input_dim']
            args['ngf']=network_args['ngf']
            args['img_size']=network_args['img_size']
            args['nc']=network_args['nc']

        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()            

    elif typ in ['ResNet18_1028','ResNet34_1028']:
        network_py= find_file_using_name(dirs='networks', model_name='resnet_cifar_1028')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()      

    elif typ in ['ResNet18_1028_R2','ResNet34_1028_R2']:
        network_py= find_file_using_name(dirs='networks', model_name='resnet_cifar_1028_R2')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()  
    

    elif typ in ['wrn_40_2_1028_R2','wrn_40_1_1028_R2','wrn_16_2_1028_R2','wrn_16_1_1028_R2']:
        network_py= find_file_using_name(dirs='networks', model_name='wresnet_1028_R2')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()        
  
    elif typ in ['vgg11_bn_1028']:
        network_py= find_file_using_name(dirs='networks', model_name='vgg_1028')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()       

    elif typ in ['vgg11_bn_1028_R2']:
        network_py= find_file_using_name(dirs='networks', model_name='vgg_1028_R2')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()    

    elif typ in ['ResNet18_1028_aap','ResNet34_1028_aap']:
        network_py= find_file_using_name(dirs='networks', model_name='resnet_cifar_1028_aap')  
        classname=typ
        args['num_classes']=network_args['num_classes']
        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()      


    elif typ in ['CMIGeneratorV3']:
        network_py= find_file_using_name(dirs='networks', model_name='cmi_generator')  
        classname=typ
        if 'input_dim' in network_args:
            args['nz']=network_args['input_dim']
            args['ngf']=network_args['ngf']
            args['img_size']=network_args['img_size']
            args['nc']=network_args['nc']

        network_method=getattr(network_py,classname)
        network_method=partial(network_method,**args)
        network=network_method()      

  

    else:
        raise(RuntimeError('...'))


    if 'pretrained_loc' in network_args and network_args['pretrained_loc'] is not None:
        pretrained_loc= network_args['pretrained_loc']
        statedict=torch.load(pretrained_loc)

        if 'part' in  network_args and network_args['part'] is not None:
            getattr(network,network_args['part']).load_state_dict(statedict)
            print('load network part %s weights using %s'%(network_args['part'],pretrained_loc))

        else:
            network.load_state_dict(statedict)
            print('load network weights using %s'%pretrained_loc)


    if 'init' in network_args and network_args['init'] is not None:
        
        if 'init_part' in network_args and network_args['init_part'] is not None:
            getattr(network, network_args['init_part']).apply(weights_init(network_args['init'])  )
            print('initial network part %s weights using %s'%( network_args['init_part'], network_args['init']) )

        else:
            network.apply(weights_init(network_args['init'])  )
            print('initial network weights using %s'%network_args['init'])
            



    return network


def find_file_using_name( dirs, model_name ): # dataset_name 默认为unaligned

    model_filename = dirs + '.' +model_name   # filename

    modelpy = importlib.import_module(model_filename)

    return modelpy