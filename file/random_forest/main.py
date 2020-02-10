'''
train a sklearn random forest model
'''
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# get env variables
TASK = os.getenv('TASK')
CKPT_FILE = os.getenv('CKPT_FILE')
EXPT_DIR = os.getenv('EXPT_DIR')
DATA_DIR = os.getenv('DATA_DIR')

# prepare data
dataset=np.genfromtxt(os.path.join(DATA_DIR, 'letter_recognition.csv'),
        header=None)

print(dataset)



