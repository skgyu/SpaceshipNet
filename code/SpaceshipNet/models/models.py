#coding: utf-8
import sys,importlib
from skyu_tools import skyu_util as sutil

def create_model(opt):

    model_cls= sutil.find_model_using_name(dirs='models', model_name=opt.model)  
    model=model_cls()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


