'''
compute mAp on test set
'''
from leanml.metrics import obj_detect_metrics
from cocodata import testset
from torch.utils.data import DataLoader
import pickle
import os

# Get test image dimension
img_size = {}; labels = {}


for i in range(len(testset)):
    size = testset[i][-1][0]
    img_size[i] = size
    label = testset[i][1]

    labels[i] = label[:, 1:].numpy().tolist()
    img_size[i] = size

# read prediction
EXPT_DIR = os.getenv('EXPT_DIR')

with open(os.path.join(EXPT_DIR, 'prediction.pkl'), 'rb') as f:
    prediction = pickle.load(f)

print(obj_detect_metrics(labels, prediction, img_size))


