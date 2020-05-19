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


import time

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

    if dev_size == 0:
        return df, None

    N = len(df)
    N_TRAIN = int(N * (1-dev_size))
    if N_TRAIN == N:
        print('Warning: dev_size too small')
        N_TRAIN = N - 1

    np.random.seed(1273162)
    rand_id = np.random.permutation(N)
    train_idx, dev_idx = rand_id[:N_TRAIN], rand_id[N_TRAIN:]
    df_tr, df_dev = df.iloc[train_idx,:], df.iloc[dev_idx,:]

    return df_tr, df_dev



# Hay dos tipos de datasets que se pueden armar con el IMDb: el 
# dataset mismo, que contiene los comentarios y sus etiquetas,
# y el dataset para entrenar modelos de lenguaje que se forma 
# de manera autosupervisada usando la hipótesis distribucional.

# Dataset supervisado: comentarios + etiquetas
class SupervisedIMDbDataset(Dataset):

    @classmethod
    def split(cls,  dev_size=.1, # Proporción del validation dataset
                    vocab=None, # Vocabulario a usar
                    test=False, # Obtener el test dataset
                    include_unk=True, # Incluír el token UNK en el vocabulario
                    include_startend=True, # Incluír los tokens START y END en el vocabulario
                    in_dtype=torch.float, # Data type de las muestras de entrada
                    out_dtype=torch.float): # Data type de las muestras de salida


        df_train, df_test, ds_unsup, full_vocab = read_dataset()
        if vocab is None:
            vocab = full_vocab
        elif isinstance(vocab,int):
            vocab = full_vocab[:vocab]

        df_train, df_dev = split_train_dev(df_train,dev_size)

        if df_dev is None and not test:
            df_dict = {'train': df_train}
        elif df_dev is None and test:
            df_dict = {'train': df_train, 'test': df_test}
        elif df_dev is not None and not test:
            df_dict = {'train': df_train, 'dev': df_dev}
        else:
            df_dict = {'train': df_train, 'dev': df_dev, 'test': df_test}
        dataset_dict = {data: cls(df,vocab,include_unk,include_startend,in_dtype,out_dtype) for data, df in df_dict.items()}
        return dataset_dict

    def __init__(self,df, vocab, include_unk=True, include_startend=True,
        in_dtype=torch.float, out_dtype=torch.long):
        self.x = None # Comentarios tokenizados y convertidos a índices
        self.y = None # Etiquetas
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)

# Dataset autosupervisado: palabra central + contexto
class SelfSupervisedIMDbDataset(Dataset):

    @classmethod
    def get_samples(cls, data, vocab=None, **kwargs):
        df_train, df_test, ds_unsup, full_vocab = read_dataset()
        if vocab is None:
            vocab = full_vocab
        elif isinstance(vocab,int):
            vocab = full_vocab[:vocab]

        if data == 'train':
            ds = df_train['comment']
        elif data == 'test':
            ds = df_test['comment']
        elif data == 'all':
            ds = df_unsup['comment']
        else:
            raise TypeError('{} no es una opción válida'.format(data))

        return cls(ds,vocab,**kwargs)

    def __init__(self,ds,vocab,**kwargs):
        self.x = None
        self.y = None

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)




class BinaryBOWDataset(SupervisedIMDbDataset):
    """
    Dataset que consiste en los comentarios del IMDb convertidos
    en un bog-of-words, y sus respectivas etiquetas en forma 
    binaria.
    """
    def __init__(self,df, vocab, include_unk=True, include_startend=True, 
        in_dtype=torch.float, out_dtype=torch.long):

        comments, idx_to_tk = utils.tokenize_dataseries(df['comment'], vocab, include_unk, include_startend)
        self.idx_to_tk = idx_to_tk
        self.x = torch.from_numpy(utils.idx_to_bow(comments, idx_to_tk, include_unk)).type(in_dtype) 
        self.y = torch.from_numpy(df['rate'].values).type(out_dtype).view(-1,1)
        self.y[self.y < 5] = 0
        self.y[self.y > 6] = 1


class CBOWDataset(SelfSupervisedIMDbDataset):
    """
    Dataset para hacer LM con el corpus de IMDb. Cada palabra del corpus
    es una etiqueta, y el contexto conformado por left_window palabras a 
    su izquierda y right_window palabras a su derecha, convertido en BOW,
    es la muestra de entrada. Este dataset es el que se usa para entrenar
    el modelo CBOW de word2vec.
    """
    def __init__(self,ds,vocab,left_window=2,right_window=2):
        # TO DO
        self.x = None
        self.y = None


class SkipGramDataset(SelfSupervisedIMDbDataset):
    """
    Dataset para hacer LM con el corpus de IMDb. Por cada palabra del corpus
    hay left_window+right_window muestras, compuestas por la palabra como 
    muestra de entrada y cada una de las palabras de su contexto como etiqueta.
    Este es el dataset utilizado para entrenar el modelo SkipGram de word2vec.
    """
    def __init__(self,ds,vocab,left_window=2,right_window=2):
        # TO DO
        self.x = None
        self.y = None


class SeqDataset(SupervisedIMDbDataset):
    """
    Dataset de secuencias supervisadas (las etiquetas son los rates)
    """
    def __init__(self,df, vocab, in_dtype=torch.float, out_dtype=torch.long):
        vocab_size = len(vocab)
        comments = utils.tokenize_dataseries(df['comment'], vocab, vocab_size)
        self.x = utils.idx_to_sequence(comments, in_dtype)
        self.y = torch.from_numpy(df['rate'].values).type(out_dtype).view(-1,1)


