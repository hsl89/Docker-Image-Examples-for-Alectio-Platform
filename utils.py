import os
import pickle

def w(expt_dir):
    '''write labeled.pkl and unlabeled.pkl files'''
    labeled = list(range(10))
    unlabeled =list(range(10, 100))

    labeled = {k: 0 for k in labeled}
    unlabeled = {k: 0 for k in unlabeled}

    fname = os.path.join(expt_dir, 'labeled.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(labeled, f)

    fname = os.path.join(expt_dir, 'unlabeled.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(unlabeled, f)
    
    return



w('/home/ubuntu/Common/tmp')
