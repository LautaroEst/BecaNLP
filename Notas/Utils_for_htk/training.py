import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
import numpy as np
import itertools

import matplotlib.pyplot as plt

class WordContextSamples(Dataset):
    
    def __init__(self, words, contexts):
        
        self.words = words
        self.contexts = contexts
        
    def __getitem__(self,idx):
        pass
    
    def __len__(self):
        return len(self.words)
    
    @classmethod
    def from_corpus(cls, corpus, left_window=2, right_window=2, split_contexts=0):
        unk_token_idx = len(corpus.vocabulary)
        context_size = left_window + right_window
        words = []
        contexts = []
        for doc in corpus.data:
            for i in range(left_window):
                doc.insert(0,unk_token_idx)
            for i in range(right_window):
                doc.append(unk_token_idx)
            for i, idx in enumerate(doc[left_window:-right_window],left_window):
                words.append(idx)
                contexts.append(doc[i-left_window:i] + doc[i+1:i+right_window+1])
            for i in range(left_window):
                doc.pop(0)
            for i in range(right_window):
                doc.pop(-1)
        
        if split_contexts == 0:
            words = torch.tensor(words)
            contexts = torch.tensor(contexts)
            mask = (words != unk_token_idx) * (contexts != unk_token_idx).any(dim=1)
        elif split_contexts == -1:
            words = torch.tensor(words).view(-1,1).repeat(1,context_size).view(-1)
            contexts = torch.tensor(contexts).view(-1)
            mask = (words != unk_token_idx) * (contexts != unk_token_idx)
        elif split_contexts < 0:
            raise RuntimeError('El tamaño del contexto debe ser positivo o igual a -1')
        elif context_size % split_contexts == 0:
            words = torch.tensor(words).view(-1,1).repeat(1,context_size // split_contexts).view(-1)
            contexts = torch.tensor(contexts).view(-1,split_contexts)
            mask = (words != unk_token_idx) * (contexts != unk_token_idx).any(dim=1)
        else:
            raise RuntimeError('Los tamaños de los contextos deben ser iguales')
        
        words = words[mask]
        contexts = contexts[mask]
        
        return cls(words, contexts)
            
    @classmethod
    def from_binary_files(cls, filenames, decode='utf-8', delimiter_pattern=' ', cutoff_freq=1, left_window=2, right_window=2, split_contexts=0):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'rb') as file:
                    texts_list.append(file.read().decode(decode))
        elif isinstance(filenames, str):
            with open(filenames, 'rb') as file:
                texts_list.append(file.read().decode(decode))
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls.from_corpus(data, cutoff_freq, left_window, right_window, split_contexts)
    
    @classmethod
    def from_text_files(cls, filenames, delimiter_pattern=' ', cutoff_freq=1, left_window=2, right_window=2, split_contexts=0):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'r') as file:
                    texts_list.append(file.read())
        elif isinstance(filenames, str):
            with open(filenames, 'r') as file:
                texts_list.append(file.read())
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls.from_corpus(data, cutoff_freq, left_window, right_window, split_contexts)
    
    @classmethod
    def from_strings(cls, texts, delimiter_pattern=' ', cutoff_freq=1, left_window=2, right_window=2, split_contexts=0):
        if isinstance(texts, list):
            texts_list = texts
        elif isinstance(filenames, str):
            texts_list = [texts]
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls.from_corpus(data, cutoff_freq, left_window, right_window, split_contexts)
    
    @classmethod
    def from_lists(cls, texts_list, cutoff_freq=1, left_window=2, right_window=2, split_contexts=0):
        data = texts_list
        return cls.from_corpus(data, cutoff_freq, left_window, right_window, split_contexts)  

    
    
class Trainer(object):
    
    def __init__(self, model, train_dataset, batch_size, device):
        
        if device is None:
            self.device = torch.device('cpu')
            self.model = model
            print('Warning: Dispositivo no seleccionado. Se utilizará la cpu.')
        elif device == 'parallelize':
            if torch.cuda.device_count() > 1:
                self.device = torch.device('cuda:0')
                self.model = nn.DataParallel(model)
            else:
                self.device = torch.device('cpu')
                self.model = model
                print('Warning: No es posible paralelizar. Se utilizará la cpu.')
        elif device == 'cuda:0' or device == 'cuda:1':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                self.model = model
            else:
                self.device = torch.device('cpu')
                print('Warning: No se dispone de dispositivos tipo cuda. Se utilizará la cpu.')
        elif device == 'cpu':
            self.device = torch.device(device)
            self.model = model
        else:
            raise RuntimeError('No se seleccionó un dispositivo válido')
        
        self.model = self.model.to(device=self.device)
        
        samples_idx = torch.randperm(len(train_dataset))
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))
        self.batch_len = len(self.dataloader)
        

    def Train(self, algorithm='SGD', epochs=1, sample_loss_every=100, **kwargs):
        
        if algorithm == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif algorithm == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **kwargs)
        self.model.train()
            
        try:
            n_iter = self.loss_history['iter'][-1]
            print('Resuming training...')
            
        except (AttributeError, IndexError): 
            print('Starting training...')
            self.loss_history = {'iter': [], 'loss': []}
            n_iter = 0
        
        print('Optimization method: {}'.format(algorithm))
        print('Learning Rate: {:.2g}'.format(kwargs['lr']))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        
            for e in range(epochs):
                for t, (x,y) in enumerate(self.dataloader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    
                    optimizer.zero_grad() # Llevo a cero los gradientes de la red
                    scores = self.model(x) # Calculo la salida de la red
                    loss = self.Loss(scores,y) # Calculo el valor de la loss
                    loss.backward() # Calculo los gradientes
                    optimizer.step() # Actualizo los parámetros

                    if (e * self.batch_len + t) % sample_loss_every == 0:
                        print('Epoch: {}, Batch number: {}, Loss: {}'.format(e+1, t,loss.item()))
                        self.loss_history['iter'].append(e * self.batch_len + t + n_iter)
                        self.loss_history['loss'].append(loss.item())
                    
            print('Training finished')
            print()            

        except KeyboardInterrupt:
            print('Exiting training...')
            print()
            self.loss_history['iter'].append(e * self.batch_len + t + n_iter)
            self.loss_history['loss'].append(loss.item())
        
        
    @staticmethod
    def Loss(scores,target):
        raise NotImplementedError
        
    
    def plot_loss_history(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self.loss_history['iter'],self.loss_history['loss'], **kwargs)
        
        return ax
    


class Word2VecTrainer(Trainer):
    
    def __init__(self, model, corpus, window_size, embedding_dim, batch_size, device):
        
        vocab_size = len(corpus.vocabulary)
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        
        if model == 'SkipGram':
            dataset = self.SkipGramSamples.from_corpus(corpus, window_size, window_size, split_contexts=-1)
            model = self.SkipGramModel(vocab_size, embedding_dim)
        elif model == 'CBOW':
            dataset = self.CBOWSamples.from_corpus(corpus, window_size, window_size, split_context=0)
            model = self.CBOWModel(vocab_size, embedding_dim)
        else:
            raise RuntimeError('Modelo seleccionado no válido.')
        
        super().__init__(model, dataset, batch_size, device)

    @staticmethod
    def Loss(scores,target):
        lf = nn.CrossEntropyLoss(reduction='mean')
        return lf(scores,target)
        
        
    class SkipGramModel(nn.Module):
        
        def __init__(self,vocab_size,embedding_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size+1,embedding_dim,padding_idx=vocab_size)
            self.out = nn.Linear(embedding_dim,vocab_size,bias=False)
            
        def forward(self,x):
            embedding = self.emb(x)
            scores = self.out(embedding)
            return scores
        
    class SkipGramSamples(WordContextSamples):
    
        def __init__(self, words, contexts):
            super().__init__(words, contexts)

        def __getitem__(self,idx):
            return self.words[idx], self.contexts[idx]
        

    class CBOWModel(nn.Module):
        
        def __init__(self,vocab_size,embedding_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size+1,embedding_dim,padding_idx=vocab_size)
            self.out = nn.Linear(embedding_dim,vocab_size,bias=False)
            
        def forward(self,x):
            embedding = self.emb(x).mean(dim=1)
            scores = self.out(embedding)
            return scores
        
    class CBOWSamples(WordContextSamples):
    
        def __init__(self, words, contexts):
            super().__init__(words, contexts)

        def __getitem__(self,idx):
            return self.contexts[idx], self.words[idx]
        