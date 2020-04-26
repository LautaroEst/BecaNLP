import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import torch
from utils import *

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
                
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--loss", "-l", help="Loss criterion")
parser.add_argument("--optim", "-o", help="Optimization algorithm")
parser.add_argument("--epochs", "-e", help="Number of epochs")
parser.add_argument("--sample_loss_every", "-s", help="Sample loss every this number")
parser.add_argument("--learning_rate", "-lr", help="Learning rate")
parser.add_argument("--check_on_train", "-cot", help="Learning rate")
args = parser.parse_args()



class IMDbDataset(Dataset):
    
    def __init__(self, samples, padding_idx=-1):
        
        num_samples = len(samples)
        lenghts = np.array([len(sample[0]) for sample in samples])
        max_len_idx = lenghts.max()
        self.x = torch.zeros(num_samples,max_len_idx, dtype=torch.long) + padding_idx
        for i, sample in enumerate(samples):
            self.x[i,:lenghts[i]] = torch.tensor(sample[0])
            
        self.y = torch.tensor([sample[1] for sample in samples], dtype=torch.float).view(-1,1)
        
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.y)
    



class BOWModel(nn.Module):
    
    def __init__(self,vocab_size, padding_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1,1,padding_idx)
        
    def forward(self,x):
        return self.emb(x).sum(dim=1)
    

def main():
    
    # Leemos y separamos en train / dev / test:
    df_train, df_dev, df_test, vocab = read_and_split_dataset()
    idx_to_tk = {idx: tk for idx, tk in enumerate(vocab)}
    vocab_size = len(vocab)

    # Tokenizamos:
    samples = {}
    unk_tk = '<UNK>'
    unk_idx = vocab_size
    for data, df in zip(['train', 'dev'],[df_train, df_dev]):
        samples[data] = tokenize_dataframe(df, vocab, unk_idx)



    padding_idx = vocab_size
    train_dataset = IMDbDataset(samples['train'],padding_idx)
    dev_dataset = IMDbDataset(samples['dev'], padding_idx)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=True)


    model = BOWModel(vocab_size, padding_idx)
    device = 'cuda:1'
    BOWTrainer = Trainer(train_loader,dev_loader,model,device)
    BOWTrainer.train(loss_fn,optim_algorithm,epochs,sample_loss_every,check_on_train,lr=learning_rate)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    loss_fn = 'BCE' if not args.loss else args.loss
    optim_algorithm = 'Adam' if not args.optim else args.optim
    epochs = 1 if not args.epochs else args.epochs
    sample_loss_every = 1 if not args.sample_loss_every else args.sample_loss_every
    check_on_train = False if not args.check_on_train else True
    learning_rate = 1e-4 if not args.learning_rate else args.learning_rate
    
    print(loss_fn, optim_algorithm, epochs, sample_loss_every, check_on_train, learning_rate)
    
    
    

