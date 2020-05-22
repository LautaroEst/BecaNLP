import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
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
        self.input_dtype = next(iter(train_dataloader))[0].dtype
        self.target_dtype = next(iter(train_dataloader))[1].dtype
        
        self.first_time = True
        self.loss_history = {'iter': [], 'loss': []}
        self.batch_len = len(self.dataloader)
        
        print('Model trainer created:')
        train_samples = int((1 - val_size) * len(train_dataset)) 
        val_samples = len(train_dataset) - train_samples
        test_samples = len(test_dataset)
        total_samples = train_samples + val_samples + test_samples
        percent_val, percent_test = (val_samples // total_samples) * 100, (test_samples // total_samples) * 100
        print('Number of training samples: {} ({}%)'.format(train_samples, 100 - percent_val - percent_test))
        print('Number of validation samples: {} ({}%)'.format(val_samples, percent_val))
        print('Number of test samples: {} ({}%)'.format(test_samples, percent_test))
        print('Number of batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()
        

    def generate_data_batches(train_dataset, test_dataset, # Train y test datasets
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


    def InitParameters(self,param_dict=None,use_gpu=None):
        
        # Defino el dispositivo sobre el cual trabajar:
        if use_gpu == 0:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        elif use_gpu == 1:
            self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        elif use_gpu is None:
            self.device = torch.device('cpu')
            
            
        if param_dict is not None:
            
            if isinstance(param_dict,dict):
                self.model.load_state_dict(param_dict)
            elif isinstance(from_pretrained,str):
                try:
                    self.model.load_state_dict({'weight':torch.load(from_pretrained)})
                    print('Embeddings loaded from file {}'.format(from_pretrained))
                except:
                    print('{} is not a file!'.format(from_pretrained))
            
            else:
                raise TypeError('from_pretrained debe ser None, nn.Embedding o torch.Tensor')
                
        else:
            self.model.emb.load_state_dict( \
            {'weight':torch.randn(self.model.emb.num_embeddings, self.model.emb.embedding_dim)})
            
        
        self.model = self.model.to(device=self.device)
        self.model.emb.weight.requires_grad = requires_grad
    
    def SGDTrain(self,
                 epochs=1,
                 learning_rate=1e-2,
                 sample_loss_every=100,
                 check_on_train=False):
        pass

    def check_accuracy(self):
        pass
                       







""" Función para calcular la cantidad de muestras predecidas correctamente:
"""
def CheckAccuracy(loader,         # Dataloader
                  model,          # Intancia del modelo 
                  device,         # Lugar en donde correr el entrenamiento (cpu o gpu)
                  input_dtype,    # Data type de las muestras de entrada
                  target_dtype):  # Data type de las muestras de salida 
    
    num_correct = 0
    num_samples = 0
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=input_dtype)  
            y = y.to(device=device, dtype=target_dtype)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        return num_correct, num_samples

    
""" Función para entrenar el modelo mediante el método del SGD:
"""
def SGDTrainModel(model,                 # Intancia del modelo
                  train_data,            # Dataloaders de entrenamiento y validación (diccionario)
                  epochs=1,              # Cantidad de epochs
                  learning_rate=1e-2,    # Tasa de aprendizaje
                  sample_loss_every=100, # Cada cuántas iteraciones calcular la loss
                  check_on_train=False,  # Calcular el accuracy en el conjunto de entrenamiento también
                  use_gpu=1):            # Usar GPUs 
    
    # Data:
    train_dataloader = train_data['train']
    val_dataloader = train_data['validation']
    
    # Data-types:
    input_dtype = next(iter(train_dataloader))[0].dtype
    target_dtype = next(iter(train_dataloader))[1].dtype
    
    # Defino el dispositivo sobre el cual trabajar:
    if use_gpu == 0:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    elif use_gpu == 1:
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    elif use_gpu is None:
        device = torch.device('cpu')
    
    
    performance_history = {'iter': [], 'loss': [], 'accuracy': []}
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    batch_size = len(train_dataloader)
    
    try:
    
        for e in range(epochs):
            for t, (x,y) in enumerate(train_dataloader):
                model.train()
                x = x.to(device=device, dtype=input_dtype)
                y = y.to(device=device, dtype=target_dtype)

                # Forward pass
                scores = model(x) 

                # Backward pass
                loss = model.loss(scores,y)                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (e * batch_size + t) % sample_loss_every == 0:
                    num_correct_val, num_samples_val = CheckAccuracy(val_dataloader, model, device, input_dtype, target_dtype)
                    performance_history['iter'].append(e * batch_size + t)
                    performance_history['loss'].append(loss.item())
                    performance_history['accuracy'].append(float(num_correct_val) / num_samples_val)
                    print('Epoch: {}, Batch number: {}'.format(e+1, t))
                    print('Accuracy on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val))
                    
                    if check_on_train:
                        num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
                        print('Accuracy on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
                        print()
                 
        return performance_history
                    
    except KeyboardInterrupt:
        
        print('Exiting training...')
        print('Final accuracy registered on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val) )
        if check_on_train:
            num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
            print('Final accuracy registered on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
            
        return performance_history

    


    
    
class Word2vecTrainer(object):
    
    """
        Clase para entrenar word embeddings. 
        Algoritmos implementados: SGD
    
    """
    
    def __init__(self,
                 corpus,                 # Corpus de entrenamiento (debe ser una lista de listas de strings).
                 cutoff_freq=1,          # Descartar palabras cuya frecuencia sea menor o igual a este valor.
                 lm='CBOW',              # Modelo de lenguaje a utilizar.
                 window_size=2,          # Tamaño de la ventana.
                 batch_size=64,          # Tamaño del batch.
                 embedding_dim=100,      # Dimensión de los word embeddings.
                 use_gpu=None):          # Flags para usar las GPUs (puede ser 0, 1 o None)
        
        self.cutoff_freq = cutoff_freq
        self.lm = lm
        self.window_size = window_size
        self.embedding_dim = embedding_dim
    
        # Defino el dispositivo sobre el cual trabajar:
        if use_gpu == 0:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        elif use_gpu == 1:
            self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        elif use_gpu is None:
            self.device = torch.device('cpu')
        
        # Obtengo los batches de muestras:
        dataset = Word2VecSamples(corpus, window_size=window_size, cutoff_freq=cutoff_freq)
        vocab_size = len(dataset.vocabulary)    
        samples_idx = torch.randperm(len(dataset))
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))

        # Defino el modelo:
        if lm == 'CBOW':
            self.model = CBOWModel(vocab_size, embedding_dim)
            self.idx = (1, 0)
        elif lm == 'SkipGram':
            self.model = SkipGramModel(vocab_size, embedding_dim)
            self.idx = (0, 1)
        else:
            raise TypeError('El modelo de entrenamiento no es válido.')

        self.first_time = True
        self.loss_history = {'iter': [], 'loss': []}
        self.batch_len = len(self.dataloader)
        
        print('Word2vec trainer created:')
        print('Model used: {}'.format(self.lm))
        print('Window size: {}'.format(window_size))
        print('Embedding dimension: {}'.format(embedding_dim))
        print('Number of samples: {}'.format(len(dataset)))
        print('Vocabulary Size: {}'.format(vocab_size))
        if cutoff_freq <= 0:
            print('No discarted words')
        else:
            print('Discarted words with frequency less than {}. Total words leaved: {}'.format(cutoff_freq, 
                                      sum([self.dataloader.dataset.vocabulary.get_freq(idx) > cutoff_freq \
                                      for idx in range(len(self.dataloader.dataset.vocabulary))])))
        print('Number of batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()

        
    def InitEmbeddings(self, from_pretrained=None, requires_grad=True):
        
        if from_pretrained is not None:
            
            if isinstance(from_pretrained,nn.Embedding):
                self.model.emb.load_state_dict(from_pretrained.state_dict())
            elif isinstance(from_pretrained,torch.Tensor):
                self.model.emb.load_state_dict({'weight':from_pretrained})
            elif isinstance(from_pretrained,str):
                try:
                    self.model.emb.load_state_dict({'weight':torch.load(from_pretrained)})
                    print('Embeddings loaded from file {}'.format(from_pretrained))
                except:
                    print('{} is not a file!'.format(from_pretrained))
            
            else:
                raise TypeError('from_pretrained debe ser None, nn.Embedding o torch.Tensor')
                
        else:
            self.model.emb.load_state_dict( \
            {'weight':torch.randn(self.model.emb.num_embeddings, self.model.emb.embedding_dim)})
            
        
        self.model = self.model.to(device=self.device)
        self.model.emb.weight.requires_grad = requires_grad
        
    def SaveEmbeddings(self,file):
        
        try:
            torch.save(self.model.emb.weight.data,file)
            print('Embeddings saved to file {}'.format(file))
        except:
            print('Embeddings could not be saved to file')
            
        
    def SGDTrain(self, epochs=1, learning_rate=1e-2, sample_loss_every=100):
        idx_x, idx_y = self.idx
        n_iter = 0 if self.first_time else self.loss_history['iter'][-1]
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        if self.first_time:
            print('Starting training...')
            self.first_time = False
        else:
            print('Resuming training...')
        
        print('Optimization method: Stochastic Gradient Descent')
        print('Learning Rate: {:.2g}'.format(learning_rate))
        print('Number of epochs: {}'.format(epochs))
        print('Running on device ({})'.format(self.device))
        print()
        
        try:
            for e in range(epochs):
                for t, sample in enumerate(self.dataloader):
                    self.model.train()
                    x = sample[idx_x].to(device=self.device, dtype=torch.long)
                    y = sample[idx_y].to(device=self.device, dtype=torch.long)
                    
                    self.optimizer.zero_grad()
                    scores = self.model(x)
                    loss = self.model.loss(scores,y)
                    loss.backward()
                    self.optimizer.step()

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
        
        embeddings = self.model.emb.weight.data
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        