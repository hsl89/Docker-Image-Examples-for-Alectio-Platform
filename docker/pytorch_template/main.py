import resnet
from dataset import cifar_train, cifar_infer, cifar_test

import random
import pickle
import copy
import sys
import os

import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.utils.data.sampler as sampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


epochs = 3

params = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 5e-4,
    }

def load(pickle_file):
    with open(pickle_file, 'rb') as f:
        dt = pickle.load(f)
    return dt

def save_dict(dictionary, saved_as):
    with open(saved_as, 'wb') as f:
        pickle.dump(dictionary, f)
        f.close()
    return

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
    if LOOP > 0:
        ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP-1))
        model.load_state_dict(torch.load(ckpt))
    
    # load selected indices
    labeled = load(os.path.join(EXPT_DIR, 'labeled.pkl'))
    indices = list(labeled.keys())

    # change dataset object's target attribute to labels
    # this is used for pollution study
    cifar_train_ = copy.deepcopy(cifar_train)

    for ix in labeled:
        cifar_train_.target[ix] = labeled[ix]
    
    # setup dataloader
    dataloader = DataLoader(cifar_train_,
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
    ckpt = os.path.join(EXPT_DIR, 'ckpt_{}.pth'.format(LOOP))
    model.load_state_dict(torch.load(ckpt))
    
    # load selected indices
    unlabeled = load(os.path.join(EXPT_DIR, 'unlabeled.pkl'))

    unlabeled = list(unlabeled.keys())

    # setup dataloader
    dataloader = DataLoader(cifar_infer,
        batch_size=64, sampler=SubsetSampler(unlabeled))
    
    proba = []
    with torch.no_grad():
        for x, _ in dataloader: # label is not needed
            x  = x.to(DEVICE)
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
    
    model = resnet.ResNet18()
    model.to(DEVICE)

    if TASK == 'train':
        train(LOOP)
    elif TASK=='test':
        validate(LOOP)
    elif TASK == 'infer':
        infer(LOOP)



