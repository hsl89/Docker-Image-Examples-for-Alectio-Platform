'''
map dataset object for pytorch model
'''
from torch.utils.data import Dataset
import os
import PIL.Image as Image

class ImageDataCLS(Dataset):
    '''Image dataset object for classification'''
    def __init__(self, root, train=True, transform=None):
        '''
        root(str): root of data directory
        train (bool): use as train set
        transform(callable) transform applied to imgs
        '''
        self.transform=transform
        self.train = train
        self.root = root
        
        if train:
            imgdir=os.path.join(root, 'train')
            labels='train_labels.txt'
        else:
            imgdir=os.path.join(root, 'val')
            labels='test_labels.txt'

        # parse label file
        self.meta_info = self.parse_meta_info(os.path.join(root, labels))

    
    def parse_meta_info(self, fpath):
        meta_info = []
        with open(fpath, 'r') as f:
            for ln in f.readlines():
                idx, image_name, wnid, label = tuple(ln.split(','))
                idx, label = int(idx), int(label)
                meta_info.append({'idx': idx, 
                    'image_name': image_name,
                    'wnid': wnid,
                    'label': label})
        return meta_info


    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, idx):
        cur = self.meta_info[idx] # current image
        if self.train:
            fpath = os.path.join(self.root, 'train', cur['wnid'],
                    'images', cur['image_name'])
        else:
            fpath = os.path.join(self.root, 'val', 'images', cur['image_name'])

        img = Image.open(fpath)


        if self.transform:
            img = self.transform(img)
        return img, cur['label'], fpath






