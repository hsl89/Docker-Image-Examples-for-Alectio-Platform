import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import time
from dataset import TEXT,entire_data 
from model import RNN
import os
from collections import Counter
import pickle

import envs
from flow import Flow


batch_size=64


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 10
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

model.embedding.weight.data.copy_(TEXT.vocab.vectors)


def train():
    epochs = 1
    ckpt = os.path.join(envs.EXPT_DIR, envs.CKPT_FILE)

    # load labeled and unlabeled dataset
    lpth = os.path.join(envs.EXPT_DIR, 'labeled.pkl')
    with open(lpth, 'rb') as f:
        labeled = pickle.load(f)

    labeled = list(labeled.keys())
    unlabeled = list(set(range(len(entire_data))) - set(labeled))
    flow = Flow(model=model, labeled=labeled, unlabeled=unlabeled,
            batch_size=batch_size, cap=None, 
            resume_from=envs.RESUME_FROM)
    
    for epoch in range(epochs):
        loss, accuracy = flow.train()
    
    # save ckpt
    ckpt = {
        'model': model.state_dict(),
        'optimizer': flow.optimizer.state_dict()
    }
    torch.save(ckpt, os.path.join(envs.EXPT_DIR, 
        envs.CKPT_FILE))

    return 

def test():
    # load labeled and unlabeled dataset
    lpth = os.path.join(envs.EXPT_DIR, 'labeled.pkl')
    with open(lpth, 'rb') as f:
        labeled = pickle.load(f)

    labeled = list(labeled.keys())
    unlabeled = list(set(range(len(entire_data))) - set(labeled))
    flow = Flow(model=model, labeled=labeled, unlabeled=unlabeled,
            batch_size=batch_size, cap=None, 
            resume_from=envs.CKPT_FILE)
    
    acc, prd, lbs = flow.test()

    # save prediction
    with open(os.path.join(envs.EXPT_DIR, 'prediction.pkl'),
            'wb') as f:
        pickle.dump(prd, f)
    return 

def infer():
    lpth = os.path.join(envs.EXPT_DIR, 'labeled.pkl')
    with open(lpth, 'rb') as f:
        labeled = pickle.load(f)

    labeled = list(labeled.keys())
    unlabeled = list(set(range(len(entire_data))) - set(labeled))
    flow = Flow(model=model, labeled=labeled, unlabeled=unlabeled,
            batch_size=batch_size, cap=None, 
            resume_from=envs.CKPT_FILE)
    
    output = flow.infer()
    # to key value pair
    output = {ix:d for ix, d in zip(unlabeled, output)}

    with open(os.path.join(envs.EXPT_DIR, 'output.pkl'), 'wb') as f:
        pickle.dump(output, f)
    return

if __name__ == '__main__':
    if envs.TASK=='train':
        train()
    elif envs.TASK=='test':
        test()
    elif envs.TASK=='infer':
        infer()





