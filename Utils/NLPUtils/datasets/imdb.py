import sys
import os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/IMDb')

import re

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from . import utils

def read_vocab():
    with open(os.path.join(DATASET_PATH,'imdb.vocab'),'r') as f:
        vocab = re.split(r'\s+',f.read())
        vocab = list(filter(lambda item: item != '',vocab))
    return vocab

def read_dataset():
    vocab = read_vocab()
    df_train = pd.read_csv(os.path.join(DATASET_PATH,'train.csv'),names=['comment','rate'],
                           dtype={'comment':str,'rate':np.int},skiprows=1)
    df_test = pd.read_csv(os.path.join(DATASET_PATH,'test.csv'),names=['comment','rate'],
                          dtype={'comment':str,'rate':np.int},skiprows=1)
    df_unsup = pd.read_csv(os.path.join(DATASET_PATH,'train_unsup_all.csv'),names=['id','comment'],
                          dtype={'id':np.int,'comment':str},skiprows=1)
    ds_unsup = df_unsup['comment']
    return df_train, df_test, ds_unsup, vocab


def split_train_dev(df,dev_size=.1):

    N = len(df)
    N_TRAIN = int(N * (1-dev_size))
    np.random.seed(1273162)
    rand_id = np.random.permutation(N)
    train_idx, dev_idx = rand_id[:N_TRAIN], rand_id[N_TRAIN:]
    df_tr, df_dev = df.iloc[train_idx,:], df.iloc[dev_idx,:]

    return df_tr, df_dev


def make_binary_labels(df):
    df.loc[df['rate'] < 5, 'rate'] = 0
    df.loc[df['rate'] > 6, 'rate'] = 1


class BinaryBOWDataset(Dataset):
    
    @classmethod
    def split(cls, dev_size=.1, vocab=None):
        df_train, df_test, ds_unsup, full_vocab = read_dataset()
        if vocab is None:
            vocab = full_vocab
        for df in [df_train, df_test]:
            make_binary_labels(df)
        df_train, df_dev = split_train_dev(df_train,dev_size)
        
        in_dtype, out_dtype = torch.float, torch.float
        df_dict = {'train': df_train, 'dev': df_dev, 'test': df_test}
        dataset_dict = {data: cls(df,vocab,in_dtype,out_dtype) for data, df in df_dict.items()}
        return dataset_dict
    
    def __init__(self,df, vocab, in_dtype=torch.float, out_dtype=torch.long):
        
        vocab_size = len(vocab)
        comments = utils.tokenize_dataseries(df['comment'], vocab, vocab_size)
        self.x = utils.idx_to_bow(comments, vocab_size, dtype=in_dtype)
        self.y = torch.from_numpy(df['rate'].values).type(out_dtype).view(-1,1)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self,idx):
        return len(self.y)


