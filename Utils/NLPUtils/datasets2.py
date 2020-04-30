import sys
import os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASETS_ROOT_PATH = os.path.join(ROOT_PATH,'Utils/Datasets')
sys.path.append(DATASETS_ROOT_PATH)

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None




class BinaryBOWIMDbDataset(Dataset):
    
    def __init__(self, samples, vocab_size):
        
        num_samples = len(samples)
        self.x = torch.zeros(num_samples,vocab_size, dtype=torch.float)
        for i, sample in enumerate(samples):
            for j in sample[0]:
                if j!= vocab_size:
                    self.x[i,j] += 1.
            
        self.y = torch.tensor([sample[1] for sample in samples], dtype=torch.float).view(-1,1)
        
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)