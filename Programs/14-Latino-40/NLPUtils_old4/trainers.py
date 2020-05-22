import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import sampler, Dataset, DataLoader

import numpy
import itertools

from .utils import *


class Trainer(object):
    
    """
        Clase madre de todos los Trainers.
    """
    
    def generate_data_batches(self,train_dataset, test_dataset, # Train y test datasets
                              batch_size = 64, # Tamaño del batch
                              val_size = .02): # Proporción de muestras utilizadas para validación 

        # Sampler
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices) 
        samples_idx = torch.randperm(len(train_dataset))
        
        if val_size != 0:
            # Separo las muestras aleatoriamente en Train y Validation:
            NUM_TRAIN = int((1 - val_size) * len(train_dataset)) 
            train_samples_idx = samples_idx[:NUM_TRAIN]
            val_samples_idx = samples_idx[NUM_TRAIN:]
            
            # Dataloader para las muestras de entrenamiento y validación:
            train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      sampler=my_sampler(train_samples_idx))
            
            val_dataloader = DataLoader(train_dataset, 
                                        batch_size=batch_size, 
                                        sampler=my_sampler(val_samples_idx))
        else:
            train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      sampler=my_sampler(samples_idx))
            val_dataloader = None
            
        if test_dataset is not None:
            # Dataloader para las muestras de testeo:
            test_dataloader = DataLoader(test_dataset, 
                                         batch_size=batch_size)
        else:
            test_dataloader = None
            
        return train_dataloader, val_dataloader, test_dataloader
    
    
    def __init__(self,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader):
        
        # Dataloaders:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        
        
    def InitModel(self, model, state_dict=None, device='cpu'):
        
        # Defino el dispositivo sobre el cual trabajar:
        if device is None:
            self.device = torch.device('cpu')
            print('No se seleccionó ningún dispositivo de entrenamiento. Se asigna la cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
            print('Dispositivo seleccionado: cpu')
        elif device == 'cuda:0' or device == 'cuda:1':
            if torch.cuda.is_available():
                self.device = torch.device(device)
                print('Dispositivo seleccionado: {}'.format(device))
            else:
                self.device = torch.device('cpu')
                print('No se dispone de GPUs. Se asigna como dispositivo de entrenamiento la cpu')
        else:
            raise TypeError('No se seleccionó un dispositivo válido')
            
        # Defino el modelo:
        self.model = model
        
        # Inicializo con los parámetros de state_dict si hubiera:
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        
        # Copio el modelo al dispositivo:
        self.model = self.model.to(device=self.device)
    
    def SaveModel(self,file):
        
        try:
            torch.save(self.model.state_dict(),file)
            print('Embeddings saved to file {}'.format(file))
        except:
            print('Embeddings could not be saved to file')
    
    def Train(self, epochs=1, sample_loss_every=100, algorithm='SGD', **kwargs):
        
        # Defino el algoritmo de optimización:
        if algorithm == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif algorithm == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **kwargs)
        self.model.train()
        
        # Identifico si es la primera vez que entreno o no:
        try:
            n_iter = self.performance_history['iter'][-1]
            print('Resuming training...')
        except (IndexError, AttributeError):
            print('Starting training...')
            self.performance_history = {'iter': [], 'loss': []}
            n_iter = 0
        
        # Varios:
        print('Optimization method: {}'.format(algorithm))
        print('Learning Rate: {:.2g}'.format(kwargs['lr']))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        # Data-types:
        input_dtype = next(iter(self.train_dataloader))[0].dtype
        target_dtype = next(iter(self.train_dataloader))[1].dtype
        
        # Comienzo a entrenar:
        batch_len = len(self.train_dataloader)
        try:
    
            for e in range(epochs):
                for t, (x,y) in enumerate(self.train_dataloader):

                    x = x.to(device=self.device, dtype=input_dtype)
                    y = y.to(device=self.device, dtype=target_dtype)

                    optimizer.zero_grad() # Llevo a cero los gradientes de la red
                    scores = self.model(x) # Calculo la salida de la red
                    loss = self.Loss(scores,y) # Calculo el valor de la loss
                    loss.backward() # Calculo los gradientes
                    optimizer.step() # Actualizo los parámetros
                    
                    if (e * batch_len + t) % sample_loss_every == 0:
                        l = loss.item()
                        print('Epoch: {}, Batch number: {}, Loss: {}'.format(e+1, t,l))
                        self.performance_history['iter'].append(e * batch_len + t + n_iter)
                        self.performance_history['loss'].append(l)
                        self.EvalPerformance()
                    
            print('Training finished')
            print()

        except KeyboardInterrupt:

            print('Exiting training...')
            print()   
            
    def Loss(self,scores,target):
        pass
    
    def EvalPerformance(self):
        pass
    
    
    
    
    
class Word2VecTrainer(Trainer):
    
    """
        Trainer para entrenar Word Embeddings.
    """
    
    def __init__(self,
                 corpus,                 # Corpus de entrenamiento (debe ser una lista de listas de strings).
                 model,                  # Tipo de modelo implementado.
                 cutoff_freq=1,          # Descartar palabras cuya frecuencia sea menor o igual a este valor.
                 window_size=2,          # Tamaño de la ventana.
                 batch_size=64):         # Tamaño del batch.
        
        self.cutoff_freq = cutoff_freq
        self.window_size = window_size
        self.method = model
        
        # Obtengo los batches de muestras:
        dataset = Word2VecSamples(corpus, model, window_size=window_size, cutoff_freq=cutoff_freq)
        dataloader, _ , _ = self.generate_data_batches(dataset, None, batch_size=batch_size, val_size=0)
        self.vocab_size = len(dataset.vocabulary)
        
        super().__init__(dataloader, None, None)
        
    class CBOWModel(nn.Module):
    
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
            self.out = nn.Linear(embedding_dim, vocab_size, bias=False)

        def forward(self,x):
            embedding = self.emb(x).mean(dim=1)
            return self.out(embedding)
        
    class SkipGramModel(nn.Module):
    
        def __init__(self, vocab_size, embedding_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
            self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
            self.vocab_size = vocab_size

        def forward(self,x):
            return self.out(self.emb(x))


    def InitModel(self, embedding_dim, state_dict=None, device='cpu'):
        
        if self.method == 'CBOW':
            model = self.CBOWModel(len(self.train_dataloader.dataset.vocabulary),embedding_dim)
            self.Loss = self.CBOWLoss
        elif self.method == 'SkipGram':
            model = self.SkipGramModel(len(self.train_dataloader.dataset.vocabulary),embedding_dim)
            self.Loss = self.SkipGramLoss
        
        super().InitModel(model, state_dict, device)
    
    
    def CBOWLoss(self,scores,target):
        lf = nn.CrossEntropyLoss(reduction='sum')
        return lf(scores,target)
    
    def SkipGramLoss(self,scores,target):
        lf = nn.CrossEntropyLoss(ignore_index=self.vocab_size,reduction='sum')
        scores = scores.view(-1,self.vocab_size,1).repeat(1,1,target.size(1))
        return lf(scores,target)
    
    def EvalPerformance(self):
        pass