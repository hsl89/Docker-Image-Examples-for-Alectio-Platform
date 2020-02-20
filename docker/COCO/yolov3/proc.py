'''
process raw inference data
'''

import pickle
import os

from leanml.core import PostProcess

EXPT_DIR = os.getenv('EXPT_DIR')

with open(os.path.join(EXPT_DIR, 'output.pkl'), 'rb') as f:
    output = pickle.load(f)

psp = PostProcess(output, 'Object Detection')

proc = psp()
print(proc)
