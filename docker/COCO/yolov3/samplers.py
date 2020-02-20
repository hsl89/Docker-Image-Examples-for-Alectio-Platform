import torch.utils.data.sampler as sampler
from collections import defaultdict
import random


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices
    
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def init_sampling(labels, size):
    '''init sampling that ensures class balance
    return at least one sample per class
    '''
    # find indices of each class
    cls_indices = defaultdict(list)
    for i, c in enumerate(labels):
        cls_indices[c].append(i)
    
    # calculate number of samples to sample per class
    num = max(size // len(cls_indices), 1)
    
    init_samples = []
    for c in cls_indices:
        init_samples.extend(random.sample(cls_indices[c], num))
    return init_samples
