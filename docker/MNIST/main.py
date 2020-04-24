from dataset import mnist_train, mnist_infer, mnist_test
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

from model import LeNet

model = LeNet()

epochs = 50

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

def train():
    # setup model
    model.train()
    model.to(DEVICE)
    
    if RESUME_FROM:
        resume_from = os.path.join(EXPT_DIR, RESUME_FROM) 
        model.load_state_dict(torch.load(resume_from))

    # load selected indices
    labeled = load(os.path.join(EXPT_DIR, 'labeled.pkl'))
    indices = list(labeled.keys())

    # change dataset object's target attribute to labels
    # this is used for pollution study
    for ix in labeled:
        mnist_train.targets[ix] = labeled[ix]

    # setup dataloader
    dataloader = DataLoader(mnist_train,
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
    ckpt = os.path.join(EXPT_DIR, CKPT_FILE)
    torch.save(model.state_dict(), ckpt)
    return

def validate():
    model.to(DEVICE)
    model.eval()
    ckpt = os.path.join(EXPT_DIR, CKPT_FILE)

    model.load_state_dict(torch.load(ckpt))
    loss_fn = nn.CrossEntropyLoss()
    dataloader = DataLoader(mnist_test, batch_size=512)
    tl, nc, pd = 0.0, 0.0, [] # total loss, number of correct prediction, prediction
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            y_ = model(x)
            loss = loss_fn(y_, y)
            pr = torch.argmax(y_, dim=1) # prediction
            pd.extend(pr.cpu().numpy().tolist())
    
    # write prediction as a pkl file
    prd = {}
    for i, p in enumerate(pd):
        prd[i]=p

    fname = os.path.join(EXPT_DIR, 'prediction.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(prd, f)
    return pd

def infer():
    model.eval()
    model.to(DEVICE)

    ckpt = os.path.join(EXPT_DIR, CKPT_FILE) 
    model.load_state_dict(torch.load(ckpt))
    
    # load selected indices
    unlabeled = load(os.path.join(EXPT_DIR, 'unlabeled.pkl'))

    unlabeled = list(unlabeled.keys())

    # setup dataloader
    dataloader = DataLoader(mnist_infer,
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
    for i, o in zip(unlabeled, x):
        output[i] = o
    
    fname = os.path.join(EXPT_DIR, 'output.pkl') # output file
    with open(fname, 'wb') as f:
        pickle.dump(output, f)
    return

if __name__ == '__main__':
    DEVICE = os.getenv('CUDA_DEVICE')
    TASK = os.getenv('TASK')
    CKPT_FILE = os.getenv('CKPT_FILE')
    RESUME_FROM = os.getenv('RESUME_FROM')
    EXPT_DIR = os.getenv('EXPT_DIR')

    if TASK == 'train':
        train()
    elif TASK=='test':
        validate()
    elif TASK == 'infer':
        infer()



