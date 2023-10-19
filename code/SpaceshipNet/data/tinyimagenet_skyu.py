from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os,pickle



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


class SkyuTinyImageNet(Dataset):

    def __init__(self,data_root,split,transform=None ):

        self.dataroot=os.path.join(data_root,'tiny-imagenet-200_skyu')
        self.split=split

        if transform is  None:
            self.transform  =  transforms.ToTensor()
        else:
            self.transform  =  transform


        train_batch_loc = os.path.join(self.dataroot, 'train' )
        val_batch_loc   = os.path.join(self.dataroot, 'val' )




        if split=='train':

            dic=unpickle(train_batch_loc)


        elif split=='val':

            dic=unpickle(val_batch_loc)
          
        else:
            raise(RuntimeError('unknown split type'))


        data=dic['data']
        labels=dic['labels']


        assert(len(data)==len(labels))

        self.len=len(data)
        self.data=data
        self.labels=labels



    def __len__(self):

        return self.len


    def __getitem__(self, idx):


        idx = idx%self.len


        img, label = self.data[idx], self.labels[idx]

        img = Image.fromarray(img)
        
        img = self.transform(img)

        label =   self.labels[ idx ]

        return   {'image': img, 'label': label, 'pos': idx}