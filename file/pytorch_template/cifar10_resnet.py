import os
import PIL.Image as Image

import random
import pickle
import copy
import sys

import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.utils.data.sampler as sampler
from torch.utils.data import Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# dataset 
class ImageDataCLS(Dataset):
    '''Image dataset object for classification'''
    def __init__(self, root, train=True, transform=None):
        '''
        root(str): root of data directory
        train (bool): use as train set
        transform(callable) transform applied to imgs
        '''
        self.transform=transform

        if train:
            imgdir=os.path.join(root, 'train')
            labels='train_labels.txt'
        else:
            imgdir=os.path.join(root, 'test')
            labels='test_labels.txt'
        # load images path and labels
        self.fpath=[]
        for im in os.listdir(imgdir):
            self.fpath.append(
                os.path.join(root, imgdir, im))
            
        self.target=[]
        with open(os.path.join(root, labels), 'r') as f:
            for ln in f.readlines():
                t = ln.split(',')[-1].strip()
                self.target.append(int(t))

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        img = Image.open(self.fpath[idx])
        t = self.target[idx]

        if self.transform:
            img = self.transform(img)
        return img, t



# model 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# setup dataset
data_dir = os.getenv('DATA_DIR') 

train_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[0, 360]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))
    ])


cifar_train = ImageDataCLS(root=data_dir,  train=True,
        transform=train_transform)
# use for infer uncertainty, dont apply transform
cifar_infer = ImageDataCLS(root=data_dir,  train=True,
        transform=test_transform)

cifar_test = ImageDataCLS(root=data_dir,  train=False, 
        transform=test_transform)



epochs = 3

params = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 5e-4,
    }



class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices
    
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def train(LOOP):
    # setup model
    model.train()
    model.to(DEVICE)

    # load model ckpt from the previous trained LOOP
    if LOOP:
        ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP-1))

        model.load_state_dict(torch.load(ckpt))
    
    # load selected indices
    labeled = os.path.join(EXPT_DIR, 'labeled.txt')

    f = open(labeled, 'r')
    indices = f.readlines()
    indices = [int(x.strip()) for x in indices]
    f.close()

    # setup dataloader
    dataloader = DataLoader(cifar_train,
        batch_size=64, sampler=SubsetRandomSampler(indices))
    
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), 
        lr=params['learning_rate'], weight_decay=params['weight_decay'])

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            y_ = model(x)
            loss = loss_fn(y_, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # save ckpt
    ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP))
    torch.save(model.state_dict(), ckpt)
    return

def validate(LOOP):
    model.eval()

    ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP))
    model.load_state_dict(torch.load(ckpt))

    loss_fn = nn.CrossEntropyLoss()

    dataloader = DataLoader(cifar_test, batch_size=512)
    tl, nc, pd = 0.0, 0.0, [] # total loss, number of correct prediction, prediction
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            y_ = model(x)
            loss = loss_fn(y_, y)
            pr = torch.argmax(y_, dim=1) # prediction
            pd.extend(pr.cpu().numpy().tolist())
    
    fname = os.path.join(EXPT_DIR, 'prediction.txt')
    f = open(fname, 'w')
    for p in pd:
        f.write('{}\n'.format(p))
    f.close()
    return pd

def infer(LOOP):
    model.eval()
    model.to(DEVICE)

    # load model ckpt from the previous trained LOOP
    if LOOP!=None:
        ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP))
        model.load_state_dict(torch.load(ckpt))
    
    # load selected indices
    unlabeled = os.path.join(EXPT_DIR, 'unlabeled.txt')

    f = open(unlabeled, 'r')
    indices = f.readlines()
    indices = [int(x.strip()) for x in indices]
    f.close()

    # setup dataloader
    dataloader = DataLoader(cifar_infer,
        batch_size=64, sampler=SubsetSampler(indices))
    
    proba = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            y_ = model(x)
            y_ = F.softmax(y_, dim=1) # explict softmax
            proba.append(y_)
    
    x = torch.cat(proba, dim=0).cpu().numpy()
    output = {}

    for i, o in zip(indices, x):
        output[i] = o
    
    fname = os.path.join(EXPT_DIR, 'output.pkl') # output file
    with open(fname, 'wb') as f:
        pickle.dump(output, f)
    return

if __name__ == '__main__':
    DEVICE = os.getenv('CUDA_DEVICE')
    TASK = os.getenv('TASK')
    LOOP = int(os.getenv('LOOP'))
    EXPT_DIR = os.getenv('EXPT_DIR')
    
    model = ResNet18()
    model.to(DEVICE)

    if TASK == 'train':
        train(LOOP)
    elif TASK=='test':
        validate(LOOP)
    elif TASK == 'infer':
        infer(LOOP)


