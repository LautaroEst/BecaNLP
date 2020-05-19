import sys
import os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASETS_ROOT_PATH = os.path.join(ROOT_PATH,'Utils/Datasets')
sys.path.append(DATASETS_ROOT_PATH)
from read_datasets import *

import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

def read_and_split_dataset():
    
    # Leemos el dataset:
    df_train, df_test, ds_unsup, vocab = read_imdb_dataset()
    
    # Separamos en train y dev:
    N = len(df_train)
    N_TRAIN = int(N * .8)
    np.random.seed(1273162)
    rand_id = np.random.permutation(N)
    train_idx, dev_idx = rand_id[:N_TRAIN], rand_id[N_TRAIN:]
    df_tr, df_dev = df_train.iloc[train_idx,:], df_train.iloc[dev_idx,:]
    
    for df in [df_tr, df_dev, df_test]:
        df.loc[df['rate'] < 4, 'rate'] = 0
        df.loc[df['rate'] > 6, 'rate'] = 1
    
    return df_tr, df_dev, df_test, vocab


def tokenize_dataframe(df, vocab, unk_idx=-1):

    tk_to_idx = {tk: idx for idx, tk in enumerate(vocab)}
    df = df.copy()
    df['comment'] = df['comment'].str.lower().apply(nltk.tokenize.word_tokenize,args=('english', False))
    samples = [([tk_to_idx.get(tk,unk_idx) for tk in row['comment']], row['rate']) for i, row in df.iterrows()]
    return samples


class Trainer(object):
    
    def __init__(self,train_loader,dev_loader,model,device='cpu'):
        
        # Seleccionamos el device:
        if device is None:
            self.device = torch.device('cpu')
            print('Warning: Dispositivo no seleccionado. Se utilizará la cpu.')
        elif device == 'parallelize':
            if torch.cuda.device_count() > 1:
                self.device = torch.device('cuda:0')
                model = nn.DataParallel(model)
            else:
                self.device = torch.device('cpu')
                print('Warning: No es posible paralelizar. Se utilizará la cpu.')
        elif device == 'cuda:0' or device == 'cuda:1':
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device('cpu')
                print('Warning: No se dispone de dispositivos tipo cuda. Se utilizará la cpu.')
        elif device == 'cpu':
            self.device = torch.device(device)
        else:
            raise RuntimeError('No se seleccionó un dispositivo válido')
            
        # Llevamos los parámetros del modelo al device:
        self.model = model.to(self.device)
        
        # Guardamos los dataloaders:
        self.train_loader = train_loader
        self.batch_len = len(train_loader)
        self.dev_loader = dev_loader
        
    
    def train(self, loss_fn='BCE', optim_algorithm='SGD', epochs=1, sample_loss_every=100, check_on_train=False, **kwargs):
        
        if loss_fn == 'BCE':
            criterion = lambda scores, target: F.binary_cross_entropy_with_logits(scores, target, reduction='mean')
        else:
            raise TypeError('Loss function not supported')
        
        if optim_algorithm == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif optim_algorithm == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **kwargs)
        self.model.train()
            
        try:
            n_iter = self.loss_history['iter'][-1]
            print('Resuming training...')
            
        except (AttributeError, IndexError): 
            print('Starting training...')
            self.loss_history = {'iter': [], 'loss': [], 'dev_acc':[], 'train_acc':[]}
            n_iter = 0
        
        print('Loss function: {}'.format(loss_fn))
        print('Optimization method: {}'.format(optim_algorithm))
        print('Learning Rate: {:.2g}'.format(kwargs['lr']))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        try:
            
            for e in range(epochs):
                for t, (x,y) in enumerate(self.train_loader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    
                    optimizer.zero_grad() # Llevo a cero los gradientes de la red
                    scores = self.model(x) # Calculo la salida de la red
                    loss = criterion(scores,y) # Calculo el valor de la loss
                    loss.backward() # Calculo los gradientes
                    optimizer.step() # Actualizo los parámetros

                    if (e * self.batch_len + t) % sample_loss_every == 0:
                        self.loss_history['iter'].append(e * self.batch_len + t + n_iter)
                        print('Epoch: {}, Batch number: {}, Loss: {}'.format(e+1, t,loss.item()))
                        self.loss_history['loss'].append(loss.item())
                        
                        num_correct, num_samples = self.CheckAccuracy('dev')
                        dev_acc = 100 * float(num_correct) / num_samples
                        print('Accuracy on validation dataset: {}/{} ({:.2f}%)'\
                              .format(num_correct, num_samples, dev_acc))
                        self.loss_history['dev_acc'].append(dev_acc)
                        
                        if check_on_train:
                            num_correct, num_samples = self.CheckAccuracy('train')
                            train_acc = 100 * float(num_correct) / num_samples
                            print('Accuracy on training dataset: {}/{} ({:.2f}%)'\
                                  .format(num_correct, num_samples, train_acc))
                            self.loss_history['train_acc'].append(train_acc)
                        print()
                    
            print('Training finished')
            print()            

        except KeyboardInterrupt:
            print('Exiting training...')
            print()
            

    def CheckAccuracy(self,dataset='dev'):
    
        if dataset == 'dev':
            loader = self.dev_loader
        elif dataset == 'train':
            loader = self.train_loader
            
        num_correct = 0
        num_samples = 0
        self.model.eval()  
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)  
                y = y.to(self.device)

                scores = self.model(x)
                #_, preds = scores.max(1)
                preds = (scores > .5).type(torch.float)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            return num_correct, num_samples

    def plot_history(self, train_acc=False, **kwargs):
        
        fig, ax = plt.subplots(1,2)
        ax[0].plot(self.loss_history['iter'],self.loss_history['loss'], **kwargs)
        ax[0].set_title('Loss')

        ax[1].plot(self.loss_history['iter'],self.loss_history['dev_acc'], label='Validation', **kwargs)
        if train_acc:
            ax[1].plot(self.loss_history['iter'],self.loss_history['train_acc'], label='Train', **kwargs)

        ax[1].set_title('Accuracy error')
        ax[1].legend()
        fig.savefig('./loss_history-{}.png'.format(now.strftime("%Y-%m-%d-%H-%M-%S")))
        
    