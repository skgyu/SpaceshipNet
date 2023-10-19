#coding: utf-8
import os,logging,sys
import torch,yaml
import importlib,time,shutil
from types import MethodType
import numpy as np
from PIL import Image
import cv2

opt=None



def upload_opt(option):
    global opt,sub
    opt=option


def get_opt():
    return opt


def makedirs(x):
    if not os.path.exists(x):
        os.makedirs(x,exist_ok=True)




def get_logger(path,name='mylogger'):

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)   


    loc=os.path.join(path) 

    f_handler = logging.FileHandler(loc)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s %(name)s %(filename)s %(message)s",
    datefmt="%Y/%m/%d %X"
    ))


    filter=logging.Filter(name)
    logger.addFilter(filter)
    
    logger.addHandler(f_handler)
    logger.propagate = False

    
    return logger



def find_model_using_name( dirs, model_name ): 

    model_filename = dirs + '.' +model_name   

    modellib = importlib.import_module(model_filename)

    model= None     
    for name, cls in modellib.__dict__.items():
        if name  == model_name:
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return model



def add_functions(model_ins, dirs ,model_name):

    assert not dirs.endswith('/')

    model_filename =    '.'.join(dirs.split('/')) + '.' +model_name   # filename

    functions_py = importlib.import_module(model_filename)


    for name ,val in functions_py.__dict__.items():
        if str(val).startswith('<function'):
            setattr(model_ins,name, MethodType(val,model_ins)  )



def log(x ,name):

    logger = logging.getLogger(name)

    logger.info(x)


def tensor2im(image_tensor, imtype=np.uint8, cent=1, factor=255./2.):


    image_numpy = image_tensor[0].cpu().float().numpy() if len(image_tensor.shape) ==4 else  image_tensor.cpu().float().numpy()
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor

    return image_numpy.astype(imtype)




def tensor2images(image_tensor, imtype=np.uint8):

    assert len(list(image_tensor.size()))==4

    images_numpy = image_tensor.cpu().float().numpy()

    
    # print(images_numpy.shape)
    images_numpy = (np.transpose(images_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0 # HWC

    return images_numpy.astype(imtype)


def save_image(image_numpy, image_path ,isgray=False,verbose=True ): #HWC

    if verbose:
        print(image_path)

    if image_numpy.shape[2]==1:
        image_numpy=image_numpy.squeeze(2)

    image_pil = Image.fromarray(image_numpy)


    if isgray:
        image_pil=image_pil.convert('L')

    image_pil.save(image_path)

def save_images(visuals,image_paths,des_dir):


    for label, image_numpys in visuals.items():
        for i,image_numpy in enumerate(image_numpys):

            image_path=image_paths[i]
            basename=os.path.split(image_path)[1]
            shortname=os.path.splitext(basename)[0]
            image_name = '%s_%s.png' % (shortname, label)   # (    epoch100_fake_p)
            save_image(image_numpy, os.path.join(des_dir,image_name))







