import sys
import os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASETS_ROOT_PATH = os.path.join(ROOT_PATH,'Utils/Datasets')
sys.path.append(DATASETS_ROOT_PATH)
from read_datasets import *

from utils import *
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", "-l", help="Loss criterion", type=str)
parser.add_argument("--optim", "-o", help="Optimization algorithm", type=str)
parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
parser.add_argument("--sample_loss_every", "-s", help="Sample loss every this number", type=int)
parser.add_argument("--learning_rate", "-lr", help="Learning rate",type=float)
parser.add_argument("--check_on_train", help="Check accuracy on train set", action="store_true")
parser.add_argument("--batch_size", "-bs", help="Number of samples per batch", type=int)
parser.add_argument("--device", "-d", help="Select CUDA or cpu device ", type=str)
parser.add_argument("--no_plot", help="Don't plot error and loss", action="store_true")
args = parser.parse_args()


def read_dataset():
    df_train, df_dev, df_test, vocab = read_and_split_dataset()
    idx_to_tk = {idx: tk for idx, tk in enumerate(vocab)}

    import nltk
    df = df_train
    df = df.copy()
    df['comment'] = df['comment'].str.lower().apply(nltk.tokenize.word_tokenize,args=('english', False))

    tk_to_freq = {tk: 0 for tk in vocab}
    for comment in df['comment']:
        for tk in comment:
            try:
                tk_to_freq[tk] += 1
            except KeyError:
                continue

    vocab_size = 10000
    freqs = np.array(list(tk_to_freq.values()))
    arg_freqs = np.argsort(freqs)[::-1]
    vocab = [vocab[i] for i in arg_freqs[:vocab_size]]
    
    samples = {}
    unk_idx = vocab_size
    for data, df in zip(['train', 'dev'],[df_train, df_dev]):
        samples[data] = tokenize_dataframe(df, vocab, unk_idx)

    return samples, vocab



class IMDbBOWDataset(Dataset):
    
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




class LogisticRegression(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size,1)
        
    def forward(self,x):
        return self.linear(x)









if __name__ == '__main__':
    
    # Obtenemos los hiperparámetros y demás configuraciones:
    print('Parsing args...')
    if not args.loss:
        raise RuntimeError('No se especificó la función de costo a utilizar')
    else:
        loss_fn = args.loss
    optim_algorithm = 'Adam' if not args.optim else args.optim
    epochs = 1 if not args.epochs else args.epochs
    sample_loss_every = 1 if not args.sample_loss_every else args.sample_loss_every
    check_on_train = False if not args.check_on_train else True
    learning_rate = 1e-4 if not args.learning_rate else args.learning_rate
    batch_size = 512 if not args.batch_size else args.batch_size
    device = 'cpu' if not args.device else args.device
    no_plot = False if not args.no_plot else True
    
    # Leer el dataset ya inspeccionado:
    print('Reading dataset...')
    samples, vocab = read_dataset()
    vocab_size = len(vocab)
    train_dataset = IMDbBOWDataset(samples['train'],vocab_size)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    dev_dataset = IMDbBOWDataset(samples['dev'],vocab_size)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=True)
    
    # Instanciar el modelo
    print('Preparing the model...')
    model = LogisticRegression(vocab_size)
    
    # Instanciar el entrenador 
    LRTrainer = Trainer(train_loader,dev_loader,model,device)
    
    # Entrenar
    LRTrainer.train(loss_fn,optim_algorithm,epochs,sample_loss_every,check_on_train,lr=learning_rate)
    if not no_plot:
        LRTrainer.plot_history(train_acc=check_on_train)
    
    
    
    
    
    