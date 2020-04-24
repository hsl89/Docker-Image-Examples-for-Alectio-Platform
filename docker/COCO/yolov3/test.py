# Model tester for image classification problems

import os
import argparse
import pickle
import shutil
import sys
from subprocess import call
import time
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, type=str,
        help='root directory for input data')
parser.add_argument('-l', '--labeled', default=10, type=int,
        help='number of labeled samples to test')
parser.add_argument('-u', '--unlabeled', default=10, type=int,
        help='number of unlabeled samples to test')

args = parser.parse_args()

class bcolors:
    '''background color'''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def _create_expt_dir():
    '''create dir to save experiment logs and model ckpts'''
    expt_dir = './.log'
    if os.path.isdir(expt_dir):
        shutil.rmtree(expt_dir)

    os.mkdir(expt_dir)
    os.environ['EXPT_DIR'] = expt_dir
    return

def _create_selected_indices_file():
    '''create file for the the model to load selcted indices
    Each sample index is mapped to a list of list, 
    where the inner list defines a bounding box. It is of the form
    [c, x, y, w, h]

    c: class
    x: normalized center x-coordinate
    y: normalized center y-coordinate
    w: normalized width
    h: normalied height
    '''
    labeled  = {}
    for ix in range(10):
        labeled[ix] = []
        for _ in range(5): # 5 objects per image
            labeled[ix].append([0, 0.1, 0.1, 0.1, 0.1])

    lbfile = os.path.join('./.log/', 'labeled.pkl')
    with open(lbfile, 'wb') as f:
        pickle.dump(labeled, f)
    return

def _setenvs(envs_dict):
    for key in envs_dict:
        os.environ[key] = envs_dict[key]
    return

def _last_read(ckpt_file):
    return time.ctime(os.stat(ckpt_file).st_atime)
    
def test_main_file():
    '''test if main.py exist'''
    if not os.path.isfile('main.py'):
        print("{} does not exist".format("main.py"))
        sys.exit()
    return

def test_init_train():
    '''test training for the initial loop'''
    _create_expt_dir()
    _create_selected_indices_file()

    _setenvs({
        "CKPT_FILE": "ckpt_0",
        "DATA_DIR": args.data,
        "TASK": "train",
        "CUDA_DEVICE": "cuda:0",
        })

    print("=> Testing on initial training with {} samples".format(args.labeled))

    call(["python", "main.py"])

    if not os.path.isfile("./.log/ckpt_0"):
        print("The required checkpoint file {} is not generated".format("ckpt_0"))
        sys.exit()
    else:
        print("You are good with initial training")

def test_intermediate_loop():
    _setenvs({
        "EXPT_DIR": "./.log/",
        "RESUME_FROM": "ckpt_0",
        "CKPT_FILE": "ckpt_1",
        "DATA_DIR": args.data,
        "TASK": "train",
        "CUDA_DEVICE": "cuda:0",
        })
    
    print("=> Testing on intermediate loops")
    call(["python", "main.py"])

    if not os.path.isfile("./.log/ckpt_1"):
        print(bcolors.FAIL + "The required checkpoint file {} is not generated".format("ckpt_1") + bclors.ENDC)
        sys.exit()
    else:
        print(bcolors.OKGREEN + "You are good with intermediate training" + bcolors.ENDC)
    
def test_test():
    '''test the test function is correctly executed'''
    _setenvs({
        "EXPT_DIR": "./.log",
        "CKPT_FILE": "ckpt_0",
        "DATA_DIR": args.data,
        "TASK": "test",
        "CUDA_DEVICE": "cuda:0",
        })

    print("=> Testing on testing")
    try:
        call(["python", "main.py"])
    except Exception as e:
        raise(e)
        
    if not os.path.isfile("./.log/prediction.pkl"):
        print(bcolors.FAIL + "The required checkpoint file {} is not generated".format("prediction.pkl") + bcolors.ENDC)
        sys.exit()
    else:
        print(bcolors.OKGREEN + "The file prediction.pkl is generated" + bcolors.ENDC)
    return

def test_eval_outfile():
    '''check the output file for evaluation is correct'''
    with open('./.log/prediction.pkl', 'rb') as f:
        pred=pickle.load(f)

    for ix in pred:
        if not isinstance(ix, int):
            raise(bcolors.FAIL + "keys of the dict saved as prediction.pkl should be of type integer got {}".format(type(ix)+ bcolors.ENDC))
        
        if not isinstance(pred[ix], list):
            raise(bcolors.FAIL + "values of the dict saved as prediction.pkl should be of type list got {}".format(type(pred[ix])+ bcolors.ENDC))
        for bbox in pred[ix]:
            if not isinstance(bbox, list):
                raise('predicted bounding box should be of type list, got {}'.format(
                    type(bbox)))
            # check if bbox is of the format <x1, y1, x2, x2, objectness, class>
            if bbox[2] <= bbox[0]:
                raise('Each bounding box should be of the format [x1, y1, x2, y2, objectness, class] + \
                        but I got {} and {} for x1 and x2. But x2 should be bigger than x1'.format(bbox[0], bbox[2]))

            if bbox[3] <= bbox[1]:
                raise('Each bounding box should be of the format [x1, y1, x2, y2, objectness, class] + \
                        but I got {} and {} for y1 and y2. But y2 should be bigger than y1'.format(bbox[1], bbox[3]))
            
            if type(bbox[4]) is not float and bbox[4] > 1.0 or bbox[4] < 0.0:
                raise('Each bounding box should be of the format [x1, y1, x2, y2, objectness, class] + \
                        but I got {} for objectness. It shoudl be a float between 0 and 1'.format(bbox[4]))

            if bbox[5] % 1.0 != 0.0:
                raise('Each bounding box should be of the format [x1, y1, x2, y2, objectness, class] + \
                        but I got {} for class. It shoudl be a float representing an integer'.format(bbox[5]))

    print(bcolors.OKGREEN + "Format of dict saved at prediction.pkl looks good" + bcolors.ENDC)

def _create_unlabeled():
    unlab = {}
    for ix in range(args.unlabeled):
        unlab[ix] = None
    with open('./.log/unlabeled.pkl', 'wb') as f:
        pickle.dump(unlab, f)
    return unlab

def test_infer():
    '''test infer function'''
    unlabeled = _create_unlabeled()

    _setenvs({
        "EXPT_DIR": "./.log/",
        "CKPT_FILE": "ckpt_0",
        "DATA_DIR": args.data,
        "TASK": "infer",
        "CUDA_DEVICE": "cuda:0",
        })

    print("=> Testing infer function")
    call(["python", "main.py"])

    if not os.path.isfile("./.log/output.pkl"):
        print(bcolors.FAIL + "The required checkpoint file {} is not generated".format("output.pkl") + bcolors.ENDC)
        sys.exit()
    else:
        print(bcolors.OKGREEN + "The file output.pkl is generated" + bcolors.ENDC)

    return


def test_infer_outfile():
    '''check the output file for inference is correct
    output.pkl is a pickled dictionary of indices of unlabeled
    samples and their boundbing boxes prediction
    
    The value of each key should be a 2d numpy array.
    The first dimension represents anchors, and the second
    dimension represents the predicted bbox
    
    Each Li corresponds to prediction of one bounding box. 
    It should be of the format
    [xc, yc, w, h, o, c1...cn]
    xc: center coordinate 
    yc: center coordinate
    w: width
    h: height
    o: objectness
    c1,...,cn: class distribution
    
    '''
    with open('./.log/unlabeled.pkl', 'rb') as f:
        unlabeled = pickle.load(f)

    with open('./.log/output.pkl', 'rb') as f:
        out=pickle.load(f)
    
    # test the keys of unlabeled and output.pkl are the same
    if set(out.keys())!= set(unlabeled.keys()):
        raise(bcolors.FAIL + "keys in the unlabeled.pkl and output.pkl are different" + bcolors.ENDC)

    for ix in out:
        if not isinstance(ix, int):
            raise(bcolors.FAIL + "keys of the dict saved as output.pkl should be of type integer got {}".format(type(ix)+ bcolors.ENDC))
        
        if not isinstance(out[ix], np.ndarray):
            raise(bcolors.FAIL + "values of the dict saved as output.pkl should be of type \
                    numpy.ndarray, I got {}".format(type(out[ix])+ bcolors.ENDC))
        
        # check all bbox on one image in parallel
        bbox = out[ix] 
        
        xc, yc, w, h = bbox[:, 0], bbox[:, 1], bbox[:,2], bbox[:,3]
        if any([xc.any() > 1.0, yc.any() > 1.0, w.any() > 1.0, h.any() > 1.0]):
            raise(bcolors.FAIL + "first 4 values of each bbox prediction\
                    should be normalized x-center, y-center, width and height\
                    , I got {}, {}, {}, {}".format(xc, yc, w, h) + bcolors.ENDC)

        obj = bbox[:,4]
        if any([(obj > 1.0).any(), (obj < 0.0).any()]):
            raise(bcolors.FAIL + "objectness at index 4 of the bounding box vector\
                    should be between 0 and 1, I got {}".format(obj) + bcolors.ENDC)

        for i, c in enumerate(bbox[:, 5:]):
            if any([c.any() > 1.0, c.any() < 0.0]):
                raise(bcolors.FAIL + "elt at index {} should be a class probability, \
                        I got {}".format(i + 5, c) + bcolors.ENDC)
            
        if np.absolute((bbox[:, 5:].sum(axis=1) - 1.0)).mean() > 0.1:
            raise Exception(bcolors.FAIL + "elts at indices 5+ should be probability distribution, I got {}, which sums to {}".format(bbox[:, 5:], bbox[:, 5:].sum(axis=1)) + bcolors.ENDC)
        
    print(bcolors.OKGREEN + "Format of inference data in output.pkl looks good" + bcolors.ENDC)

    


if __name__ == '__main__':
    # first stage
    test_main_file()
    test_init_train()
    test_intermediate_loop()
    test_test()
    test_eval_outfile()

    test_infer()
    test_infer_outfile()


