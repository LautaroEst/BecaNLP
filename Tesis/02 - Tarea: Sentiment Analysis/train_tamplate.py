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
    pass



class MyDataset(Dataset):
    
    def __init__(self):
        pass
        
    def __getitem__(self,idx):
        pass
    
    def __len__(self):
        pass







class MyModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self,x):
        return x









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
    read_dataset()
    train_dataset = MyDataset()
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    dev_dataset = MyDataset()
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=True)
    
    # Instanciar el modelo
    print('Preparing the model...')
    model = MyModel()
    
    # Instanciar el entrenador 
    MyTrainer = Trainer(train_loader,dev_loader,model,device)
    
    
    
    
    # Entrenar
    MyTrainer.train(loss_fn,optim_algorithm,epochs,sample_loss_every,check_on_train,lr=learning_rate)
    if not no_plot:
        MyTrainer.plot(train_acc=check_on_train)
    
    
    
    
    
    