import torch
from torch.utils.data import Dataset

import pandas as pd





class NLPDataset(Dataset):
    
    def __init__(self, x_ds, y_ds, vocabulary):
        pass
    
    @classmethod
    def from_dataframe(cls, df, vocab_words=None):
        pass
    
    @classmethod
    def from_csv(cls, filename):
        pass
    
    def text_cleanning(self):
        pass
        
    