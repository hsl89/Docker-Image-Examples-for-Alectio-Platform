from model import Darknet
from cocodata import trainset, testset
from hyperparameters import hyp
from flow import Flow
import torch
from utils.utils import *
import os

# get envs

DEVICE = os.getenv('CUDA_DEVICE')
TASK = os.getenv('TASK')
CKPT_FILE = os.getenv('CKPT_FILE')
RESUME_FROM = os.getenv('RESUME_FROM')
EXPT_DIR = os.getenv('EXPT_DIR')
CUDA_DEVICE = os.getenv('CUDA_DEVICE')


def train():
    epochs=1
    model = Darknet()
    
    if RESUME_FROM:
        ckpt = torch.load(os.path.join(EXPT_DIR, RESUME_FROM))
        model.load_state_dict(ckpt['model'])

    trainset.augment=True
    fw = Flow(model, trainset, testset, hyp)
    
    with open(os.path.join(EXPT_DIR, 'labeled.pkl'), 'rb') as f:
        selected = pickle.load(f)

    # if want to try pollution study
    # load labels from EXPT_DIR/labeled.pkl
    
    for epoch in range(epochs):
        fw.train(epoch, samples=list(selected.keys()),
                prebias=False)
    

    return

def test():
    model = Darknet()
    print('Size of the test set:{}'.format(len(testset)))

    # load ckpt
    ckpt = torch.load(os.path.join(EXPT_DIR, CKPT_FILE))
    model.load_state_dict(ckpt['model'])
    fw = Flow(model, trainset, testset, hyp)
    # write prediction
    fw.validate(batch_size=16)
    
def infer():
    model = Darknet()
    ckpt = torch.load(os.path.join(EXPT_DIR, CKPT_FILE))
    model.load_state_dict(ckpt['model'])
    fw = Flow(model, trainset, testset, hyp)
    
    # get the indices of unlabeled data
    with open(os.path.join(EXPT_DIR, 'unlabeled.pkl'), 'rb') as f:
        dt = pickle.load(f)
    unlabeled = [i for i in dt]

    fw.infer(unlabeled)
    return

if __name__=='__main__':
    if TASK=='train':
        train()
    elif TASK=='test':
        test()
    elif TASK=='infer':
        infer()

