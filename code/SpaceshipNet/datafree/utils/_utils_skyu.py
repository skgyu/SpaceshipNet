import torch
from torch.utils.data import ConcatDataset, Dataset
import numpy as np 
from PIL import Image
import os, random, math
from copy import deepcopy
from contextlib import contextmanager
import lmdb
import pyarrow as pa
from data.lmdbDataset0809  import LMDBFeatureLabelImageDataset0809
from data.lmdbDataset1024  import LMDBFeatureLabelImageDataset1024
import pickle5,pickle
import time
import gc
from skyu_tools import skyu_util as sutil 
    
def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))




def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images


def skyu_collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):

    last_time=0
    images = []
    last_batch_images=[]

    if isinstance( postfix, str):
        postfix = [ postfix ]

    for dirpath, dirnames, files in os.walk(root):
        #for pos in postfix:
        for f in files:
            f_end=f.split('.')[-1]



            if f_end in postfix:
                images.append( os.path.join( dirpath, f ) )

                generate_time= int(os.path.split(f)[-1].split('-')[0])

                if generate_time>last_time:
                    last_time=generate_time
                    last_batch_images=[]
                    last_batch_images.append( os.path.join( dirpath, f ) )

                elif generate_time==last_time:
                    last_batch_images.append( os.path.join( dirpath, f ) )

    return images,last_batch_images






def compare_path(x):
    
    x= os.path.split(x)[-1].split('.')[0]
    a,b=x.split('-')
    a=int(a)
    b=int(b)
    return a,b




class SkyuFeatureLabelImagePool0809(object):

    def __init__(self, root, name='feature_label_images.lmdb'):

        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        lmdb_path= os.path.join(self.root, name)

        self.db = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
            map_size=1099511627776 * 2, readonly=False,
            meminit=False, map_async=True, lock=False)    


        db=self.db

        with db.begin(write=False) as txn:
            stat_dic=txn.stat()
            n_entries=stat_dic['entries']

        if n_entries>0:
            with db.begin(write=False) as txn:
                self.length =pickle5.loads(txn.get(b'__len__'))
                self.keys= pickle5.loads(txn.get(b'__keys__'))
                self._idx= pickle5.loads(txn.get(b'_idx'))
        else:
            self._idx = 0
            self.keys=[]
            self.length=0

        self.lmdb_path=lmdb_path


        savedir_path= os.path.join(self.root, name.split('.')[0])
        self.saveimg_path= os.path.join(savedir_path, 'images')
        self.savefeature_path= os.path.join(savedir_path, 'features')

        os.makedirs(self.saveimg_path, exist_ok=True)
        os.makedirs(self.savefeature_path, exist_ok=True)

        self.dataset=None


    def lmdb_save_feature_label_image_batch(self,features, labels, images, saveprefix):


        db=self.db

        txn=db.begin(write=True)


        for idx, fea in enumerate(features):
            savename = saveprefix+'-%d'%(idx)
            

            txn.put(savename.encode('ascii'), pickle5.dumps(   labels[idx]   ,  protocol=5)  )


            savename_np=  savename+'.npy'

            savename_np_img_loc=  os.path.join(self.saveimg_path, savename_np)
            np.save(savename_np_img_loc, images[idx])

            savename_np_feature_loc=  os.path.join(self.savefeature_path, savename_np)
            np.save(savename_np_feature_loc, features[idx])


            self.length+=1
            self.keys.append(savename)
        txn.commit()

        self._idx+=1


        assert(self.length==len(self.keys))

        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle5.dumps(self.keys,  protocol=5))
            txn.put(b'__len__', pickle5.dumps(self.length,  protocol=5) )
            txn.put(b'_idx', pickle5.dumps(self._idx,  protocol=5) )

        db.sync()

    def add(self, features, labels, images, targets=None):

        self.lmdb_save_feature_label_image_batch(features, labels, images, "%d"%(self._idx) )
        

    def get_dataset(self):

        if self.dataset is not None:

            del self.dataset 

            gc.collect()

        self.dataset = LMDBFeatureLabelImageDataset0809(db=self.db, saveimg_path = self.saveimg_path,  savefeature_path = self.savefeature_path , do_norm_range= False)

        return self.dataset



class SkyuFeatureLabelImagePool1024(object):

    def __init__(self, root, name='feature_label_images.lmdb'):

        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self._idx = 0
        self.keys=[]
        self.length=0


        savedir_path= os.path.join(self.root, name.split('.')[0])

        self.dataset=None

        self.opt = sutil.get_opt()
        opt=self.opt

        n_epoch = opt.max_epoch+1
        batch_size = opt.batch_size

        self.all_labels = [[None] * (batch_size+5) ] * (n_epoch+5)
        self.all_images = [[None] * (batch_size+5) ] * (n_epoch+5)
        self.all_features = [[None] * (batch_size+5) ] * (n_epoch+5)



    def lmdb_save_feature_label_image_batch(self,features, labels, images, epochidx):


        nfeature=len(features)

        for idx in range(nfeature):
            savename = '%d-%d'%(epochidx, idx)
            
            self.all_labels[epochidx][idx] =  labels[idx]


            self.all_images[epochidx][idx] = images[idx]


            self.all_features[epochidx][idx] = features[idx]


            self.length+=1
            self.keys.append(savename)

        self._idx+=1

        

        assert(self.length==len(self.keys))



    def add(self, features, labels, images, targets=None):

        self.lmdb_save_feature_label_image_batch(features, labels, images, self._idx )
        

    def get_dataset(self):


        if self.dataset is not None:

            del self.dataset 

            gc.collect()

        db = {'__keys__':  self.keys,
            '__len__': self.length,
            '_idx': self._idx,
            'labels':self.all_labels, 
            'images':self.all_images, 
            'features':self.all_features }

        self.dataset = LMDBFeatureLabelImageDataset1024(db= db , do_norm_range= False)

        return self.dataset




class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass




