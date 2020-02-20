from utils.datasets import LoadImagesAndLabels
from hyperparameters import hyp
import os
import pickle

DATA_DIR = os.getenv('DATA_DIR')

EXPT_DIR = os.getenv('EXPT_DIR')

# get indices of selected data and its corresponding labels
with open(os.path.join(EXPT_DIR, 'labeled.pkl'), 'rb') as f:
    labeled = pickle.load(f)

# parse the train data info

def parse_datainfo(lbfile='train_labels.txt'):
    data_info = {}
    with open(os.path.join(DATA_DIR, lbfile), 'r') as f:
        for ln in f.readlines():
            ln = ln.strip()
            idx, imgpth, lbpth = ln.split(',')
            data_info[int(idx)] = {
                'imgpth': imgpth,
                'lbpth': lbpth
                }
    return data_info

train_data = parse_datainfo()

test_data = parse_datainfo('test_labels.txt')


trainset = LoadImagesAndLabels(
    DATA_DIR,
    train_data,
    img_size=416,
    batch_size=16,
    hyp=hyp,
    augment=True,
    rect=False,
    single_cls=False)

testset = LoadImagesAndLabels(
    DATA_DIR,
    test_data,
    img_size=416,
    batch_size=16,
    hyp=hyp,
    augment=False,
    rect=False,
    single_cls=False)

if __name__ == '__main__':
    for i in range(10):
        im, tar, paths, shape = trainset[i]
        print(im.shape, paths)    
