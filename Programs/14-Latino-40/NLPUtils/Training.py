import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import torch.nn as nn
from .WordVectors import *
import numpy as np


class ModelTrainer(object):
    
    def __init__(self,
                 train_dataset,
                 test_dataset,
                 batch_size=64,
                 val_size=.02):
        
        # Data:
        tr, val, te = self.generate_data_batches(train_dataset, test_dataset,batch_size,val_size)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = tr, val, te
        
        # Data-types:
        self.input_dtype = next(iter(self.train_dataloader))[0].dtype
        self.target_dtype = next(iter(self.train_dataloader))[1].dtype
        
        self.first_time = True
        self.batch_len = len(self.train_dataloader)
        
        print('Model trainer created:')
        train_samples = int((1 - val_size) * len(train_dataset)) 
        val_samples = len(train_dataset) - train_samples
        test_samples = len(test_dataset)
        total_samples = train_samples + val_samples + test_samples
        percent_val, percent_test = int((val_samples / total_samples) * 100), int((test_samples / total_samples) * 100)
        print('Number of training samples: {} ({}%)'.format(train_samples, 100 - percent_val - percent_test))
        print('Number of validation samples: {} ({}%)'.format(val_samples, percent_val))
        print('Number of test samples: {} ({}%)'.format(test_samples, percent_test))
        print('Number of train batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()
        
        
    def generate_data_batches(self,train_dataset, test_dataset, # Train y test datasets
                              batch_size = 64, # Tamaño del batch
                              val_size = .02): # Proporción de muestras utilizadas para validación 
    
        """
            Función para iterar sobre los batches de muestras. 
            Devuelve los dataloaders de train / validation / test.
            
        """

        # Separo las muestras aleatoriamente en Train y Validation:
        NUM_TRAIN = int((1 - val_size) * len(train_dataset)) 
        samples_idx = torch.randperm(len(train_dataset))
        train_samples_idx = samples_idx[:NUM_TRAIN]
        val_samples_idx = samples_idx[NUM_TRAIN:]
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices) # sampler

        # Dataloader para las muestras de entrenamiento:
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      sampler=my_sampler(train_samples_idx))

        # Dataloader para las muestras de validación:
        val_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    sampler=my_sampler(val_samples_idx))

        # Dataloader para las muestras de testeo:
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader
    
    
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
        
        
    def Train(self, algorithm='SGD', epochs=1, sample_loss_every=100, check_on_train=False, **kwargs):
        
        if algorithm == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif algorithm == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **kwargs)
        self.model.train()
        
        if self.first_time:
            print('Starting training...')
            self.performance_history = {'iter': [], 'loss': [], 'accuracy': []}
            n_iter = 0
            self.first_time = False
        else:
            n_iter = self.loss_history['iter'][-1]
            print('Resuming training...')
        
        print('Optimization method: {}'.format(algorithm))
        print('Learning Rate: {:.2g}'.format(kwargs['lr']))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        try:
    
            for e in range(epochs):
                for t, (x,y) in enumerate(self.train_dataloader):

                    x = x.to(device=self.device, dtype=self.input_dtype)
                    y = y.to(device=self.device, dtype=self.target_dtype)

                    optimizer.zero_grad() # Llevo a cero los gradientes de la red
                    scores = self.model(x) # Calculo la salida de la red
                    loss = self.model.loss(scores,y) # Calculo el valor de la loss
                    loss.backward() # Calculo los gradientes
                    optimizer.step() # Actualizo los parámetros

                    if (e * self.batch_len + t) % sample_loss_every == 0:
                        num_correct_val, num_samples_val = self.check_accuracy('validation')
                        self.performance_history['iter'].append(e * self.batch_len + t + n_iter)
                        self.performance_history['loss'].append(loss.item())
                        self.performance_history['accuracy'].append(float(num_correct_val / num_samples_val))
                        print('Epoch: {}, Batch number: {}'.format(e+1, t))
                        print('Accuracy on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val))
                        print()

                        if check_on_train:
                            num_correct_train, num_samples_train = self.check_accuracy('train')
                            print('Accuracy on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
                            print()

            print('Training finished')
            print()

        except KeyboardInterrupt:

            print('Exiting training...')
            print()    

    def check_accuracy(self, dataset='validation'):
        
        num_correct = 0
        num_samples = 0
        
        if dataset == 'train':
            loader = self.train_dataloader
        elif dataset == 'validation':
            loader = self.val_dataloader
        elif dataset == 'test':
            loader = self.test_dataloader
        else:
            raise AttributeError('Please specify on which dataset to perform de accuracy calculation')
        
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=self.input_dtype)  
                y = y.to(device=self.device, dtype=self.target_dtype)

                scores = self.model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

        self.model.train()
        return num_correct, num_samples

    def CheckResultsOnTest(self):
        
        total_corrects = 0
        total_samples = 0
        total_performance = 0.
        
        for (x,y) in enumerate(self.test_dataloader):
            x = x.to(device=self.device, dtype=self.input_dtype)
            y = y.to(device=self.device, dtype=self.target_dtype)
            num_correct, num_samples = self.check_accuracy('test')
            total_corrects += num_corrects
            total_samples += num_samples
            total_performance += float(num_correct / num_samples)
        
        print('Final accuracy on test set: {}/{} ({}%)'.format(total_corrects,total_samples,total_performance))


class Word2vecTrainer(object):
    
    """
        Clase para entrenar word embeddings. 
    
    """
    
    def __init__(self,
                 corpus,                 # Corpus de entrenamiento (debe ser una lista de listas de strings).
                 cutoff_freq=1,          # Descartar palabras cuya frecuencia sea menor o igual a este valor.
                 window_size=2,          # Tamaño de la ventana.
                 batch_size=64):         # Tamaño del batch.
        
        self.cutoff_freq = cutoff_freq
        self.window_size = window_size
        
        # Obtengo los batches de muestras:
        dataset = Word2VecSamples(corpus, window_size=window_size, cutoff_freq=cutoff_freq)
        samples_idx = torch.randperm(len(dataset))
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))
        self.batch_len = len(self.dataloader)
        
        self.vocab_size = len(dataset.vocabulary)    
        self.first_time = True
        
        print('Word2vec trainer created:')
        print('Window size: {}'.format(window_size))
        print('Number of samples: {}'.format(len(dataset)))
        print('Vocabulary Size: {}'.format(self.vocab_size))
        print('Number of batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()

        
    def InitModel(self, model='CBOW', state_dict=None, device='cpu', paralelize=False, **kwargs):
        
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
        try:
            self.embedding_dim = kwargs['embedding_dim']
        except KeyError:
            print('Dimensión del espacio de los embeddings seleccionada automáticamente en 100.')
            self.embedding_dim = 100
        print('Dimensión del espacio de los embeddings: {}'.format(self.embedding_dim))
        
        if model == 'CBOW':
            self.model = CBOWModel(self.vocab_size, self.embedding_dim)
            self.idx = (1, 0)
        elif model == 'SkipGram':
            self.model = SkipGramModel(self.vocab_size, self.embedding_dim)
            self.idx = (0, 1)
        else:
            raise TypeError('El modelo de entrenamiento no es válido.')
        
        # Inicializo con los parámetros de state_dict si hubiera:
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        
        # Copio el modelo al dispositivo:
        if torch.cuda.device_count() > 1 and paralelize:
            self.model = nn.DataParallel(self.model)
            self.loss_fn = self.model.module.loss
        else:
            self.loss_fn = self.model.loss
        self.model = self.model.to(device=self.device)
        
    def SaveModel(self,file):
        
        try:
            torch.save(self.model.state_dict(),file)
            print('Embeddings saved to file {}'.format(file))
        except:
            print('Embeddings could not be saved to file')
            
        
    def Train(self, algorithm='SGD', epochs=1, sample_loss_every=100, **kwargs):
        
        idx_x, idx_y = self.idx
        
        if algorithm == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **kwargs)
        elif algorithm == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **kwargs)
        self.model.train()
        
        if self.first_time:
            print('Starting training...')
            self.loss_history = {'iter': [], 'loss': []}
            n_iter = 0
            self.first_time = False
        else:
            n_iter = self.loss_history['iter'][-1]
            print('Resuming training...')
        
        print('Optimization method: {}'.format(algorithm))
        print('Learning Rate: {:.2g}'.format(kwargs['lr']))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        try:
            for e in range(epochs):
                for t, sample in enumerate(self.dataloader):
                    x = sample[idx_x].to(device=self.device)
                    y = sample[idx_y].to(device=self.device)
                    
                    optimizer.zero_grad() # Llevo a cero los gradientes de la red
                    scores = self.model(x) # Calculo la salida de la red
                    loss = self.loss_fn(scores,y) # Calculo el valor de la loss
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
            
            
            
    def GetCloseVectors(self, word_list, firsts=10):
        
        embeddings = self.model.module.emb.weight.data
        vocab = self.dataloader.dataset.vocabulary
        distance = torch.nn.CosineSimilarity()

        print('Word\t\t\tClosest Words\t\t\tCosine Distance')
        print('-' * 71)

        for word in word_list:
            word_emb = embeddings[vocab.token_to_index(word),:]
            dist = distance(embeddings,word_emb.view(1,-1).repeat(len(vocab)+1,1))
            dist_idx = torch.argsort(dist,descending=True)

            cw = vocab.index_to_token(dist_idx[1].item())
            if len(word) > 7:
                if len(cw) > 7:
                    print('{}\t\t{}\t\t\t{:4f}'.format(word,vocab.index_to_token(dist_idx[1].item()),dist[dist_idx[1]]))
                else:
                    print('{}\t\t{}\t\t\t\t{:4f}'.format(word,vocab.index_to_token(dist_idx[1].item()),dist[dist_idx[1]]))
            else:
                if len(cw) > 7:
                    print('{}\t\t\t{}\t\t\t{:4f}'.format(word,vocab.index_to_token(dist_idx[1].item()),dist[dist_idx[1]]))
                else:
                    print('{}\t\t\t{}\t\t\t\t{:4f}'.format(word,vocab.index_to_token(dist_idx[1].item()),dist[dist_idx[1]]))

            for i in range(2,firsts+1):
                cw = vocab.index_to_token(dist_idx[i].item())
                if len(cw) > 7:
                    print('\t\t\t{}\t\t\t{:4f}'.format(cw,dist[dist_idx[i]]))
                else:
                    print('\t\t\t{}\t\t\t\t{:4f}'.format(cw,dist[dist_idx[i]]))

            print()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        