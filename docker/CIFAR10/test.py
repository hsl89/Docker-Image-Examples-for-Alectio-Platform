# Model tester for image classification problems

import os
import argparse
import pickle
import shutil
import sys
from subprocess import call
import time


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
    '''create file for the the model to load selcted indices'''

    labeled = {ix: 0 for ix in range(args.labeled)}

    lbfile = os.path.join('./.log', 'labeled.pkl')
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
        "EXPT_DIR": "./.log/",
        "CKPT_FILE": "ckpt_0",
        "DATA_DIR": args.data,
        "TASK": "test",
        "CUDA_DEVICE": "cuda:0",
        })

    print("=> Testing on testing")
    call(["python", "main.py"])
    
    
    if not os.path.isfile("./.log/prediction.pkl"):
        print(bcolors.FAIL + "The required checkpoint file {} is not generated".format("prediction.pkl") + bcolors.ENDC)
        sys.exit()
    else:
        print(bcolors.OKGREEN + "The file prediction.pkl is generated" + bcolors.ENDC)
    
def test_eval_outfile():
    '''check the output file for evaluation is correct'''
    with open('./.log/prediction.pkl', 'rb') as f:
        pred=pickle.load(f)

    for ix in pred:
        if not isinstance(ix, int):
            raise(bcolors.FAIL + "keys of the dict saved as prediction.pkl should be of type integer got {}".format(type(ix)+ bccolors.ENDC))
        
        if not isinstance(pred[ix], int):
            raise(bcolors.FAIL + "values of the dict saved as prediction.pkl should be of type integer got {}".format(type(pred[ix])+ bccolors.ENDC))
        
    print(bcolors.OKGREEN + "Format of dict saved at prediction.pkl looks good" + bcolors.OKGREEN)


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
    '''check the output file for inference is correct'''
    with open('./.log/unlabeled.pkl', 'rb') as f:
        unlabeled = pickle.load(f)

    with open('./.log/output.pkl', 'rb') as f:
        out=pickle.load(f)
    
    for ix in out:
        if not isinstance(ix, int):
            raise(bcolors.FAIL + "keys of the dict saved as output.pkl should be of type integer got {}".format(type(ix)+ bccolors.ENDC))
        
        if not isinstance(out[ix], list):
            raise(bcolors.FAIL + "values of the dict saved as output.pkl should be of type integer got {}".format(type(out[ix])+ bccolors.ENDC))

    
    # test the keys of unlabeled and output.pkl are the same
    if set(out.keys())!= set(unlabeled.keys()):
        raise(bcolors.FAIL + "keys in the unlabeled.pkl and output.pkl are different" + bcolors.ENDC)

    # check the value of out if a list of float
    for ix in out:
        for prob in out[ix]:
            if not isinstance(prob, float):
                raise(bcolors.FAIL + "value for each key in output.pkl should be a list of float, but some element in the value of {} is {}".format(ix, type(prob)) + bcolors.ENDC)

    
        
    print(bcolors.OKGREEN + "Format of dict saved at output.pkl looks good" + bcolors.ENDC)



if __name__ == '__main__':
    test_main_file()
    test_init_train()
    test_test()
    test_eval_outfile()
    test_infer()
    test_infer_outfile()
    


