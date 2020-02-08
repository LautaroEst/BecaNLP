import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
import numpy as np
import itertools

class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, tokens_dict={}, frequencies_dict={}):
        
        self._idx_to_tk = tokens_dict
        self._tk_to_idx = {tk: idx for idx, tk in tokens_dict.items()}
        self._idx_to_freq = frequencies_dict
        self.max_idx = len(self)
        
    @classmethod
    def from_corpus(cls, corpus, cutoff_freq=0):
        corpus_words = sorted(list(set([item for sublist in corpus for item in sublist])))
        freqs_dict = {word: 0 for word in corpus_words}
        for doc in corpus:
            for token in doc:
                freqs_dict[token] += 1
        freqs = np.array(list(freqs_dict.values()))
        mask = freqs > cutoff_freq
        corpus_words = {idx: tk for idx, tk in enumerate(itertools.compress(corpus_words,mask))}
        freqs = {idx: freq for idx, freq in enumerate(freqs[mask])}
        return cls(corpus_words, freqs)

    def index_to_token(self, index):
        return self._idx_to_tk[index]

    def token_to_index(self, token):
        return self._tk_to_idx[token]
        
    def get_freq(self, tk_or_idx):
        
        if isinstance(tk_or_idx, int):
            freq = self._idx_to_freq[tk_or_idx]
        elif isinstance(tk_or_idx, str):
            freq = 0 if tk_or_idx not in self._tk_to_idx else self._idx_to_freq[self._tk_to_idx[tk_or_idx]]
        else:
            raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        return freq

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._idx_to_tk)
    
    def __getitem__(self,tk_or_idx):
        if isinstance(tk_or_idx, int):
            return self.index_to_token(tk_or_idx)
        if isinstance(tk_or_idx, str):
            return self.token_to_index(tk_or_idx)
        raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= self.max_idx:
            raise StopIteration
        else:
            token = self._idx_to_tk[self.current]
            self.current += 1
            return token

    def __contains__(self,key):
        return key in self._tk_to_idx
    

def samples_generator(doc, vocabulary, window_size):
    for t, token in enumerate(doc):
        if token in vocabulary:
            len_doc = len(doc)
            cond1 = max(-1,t-window_size) == -1
            cond2 = min(t+window_size, len_doc) == len_doc
            if cond1 and cond2:
                context = itertools.chain(doc[:t],doc[t+1:])
            if cond1 and not cond2:
                context = itertools.chain(doc[:t],doc[t+1:t+window_size+1])
            if cond2 and not cond1:
                context = itertools.chain(doc[t-window_size:t],doc[t+1:])
            if not cond1 and not cond2:
                context = itertools.chain(doc[t-window_size:t],doc[t+1:t+window_size+1])

            context_list = [vocabulary.token_to_index(tk) for tk in context if tk in vocabulary]
            if len(context_list) != 0:
                yield (vocabulary.token_to_index(token), context_list)

                
                
class CBOWSamples(Dataset):
    
    unk_token = '<UNK>'
    
    def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
        # Obtengo el vocabulario a partir del corpus ya tokenizado:
        self.vocabulary = Vocabulary.from_corpus(corpus,cutoff_freq=cutoff_freq)
    
        # Obtengo el contexto a partir del corpus:
        self.padding_idx = len(self.vocabulary)
        self.window_size = window_size
        
        word_indeces = []
        word_contexts = []
        for doc in corpus:
            gen = samples_generator(doc, self.vocabulary, window_size)
            for word_index, word_context in gen:
                word_indeces.append(word_index)
                padd_num = 2 * window_size - len(word_context)
                if padd_num > 0:
                    word_contexts.append(word_context + [self.padding_idx for i in range(padd_num)])
                else:
                    word_contexts.append(word_context)
        
        self.word_indeces = torch.tensor(word_indeces,dtype=torch.long)
        self.context_indeces = torch.tensor(word_contexts,dtype=torch.long)
        
    def __getitem__(self,idx):
        return self.context_indeces[idx,:], self.word_indeces[idx]
    
    def __len__(self):
        return len(self.word_indeces)
            
                
class CBOWTrainer(object):
    
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
        dataset = CBOWSamples(corpus, window_size=window_size, cutoff_freq=cutoff_freq)
        samples_idx = torch.randperm(len(dataset))
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))
        self.batch_len = len(self.dataloader)
        
        self.vocab_size = len(dataset.vocabulary)    
        
        print('CBOW trainer created:')
        print('Window size: {}'.format(window_size))
        print('Number of samples: {}'.format(len(dataset)))
        print('Vocabulary Size: {}'.format(self.vocab_size))
        print('Number of batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()

        
    def InitModel(self, state_dict=None, device='cpu', paralelize=False, **kwargs):
        
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
        
        self.model = CBOWModel(self.vocab_size, self.embedding_dim)
        
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
        
        try:
            for e in range(epochs):
                for t, (x,y) in enumerate(self.dataloader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    
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
        
    
class CBOWModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self,x):
        embedding = self.emb(x).mean(dim=1)
        return self.out(embedding)
    
    def loss(self,scores,target):
        lf = nn.CrossEntropyLoss(reduction='sum')
        return lf(scores,target)
        
        

class SkipGramSamples(Dataset):
    
    unk_token = '<UNK>'
    
    def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
        # Obtengo el vocabulario a partir del corpus ya tokenizado:
        self.vocabulary = Vocabulary.from_corpus(corpus,cutoff_freq=cutoff_freq)
    
        # Obtengo el contexto a partir del corpus:
        self.padding_idx = len(self.vocabulary)
        self.window_size = window_size
        
        word_indeces = []
        word_contexts = []
        for doc in corpus:
            gen = samples_generator(doc, self.vocabulary, window_size)
            for word_index, word_context in gen:
                word_indeces.append(word_index)
                padd_num = 2 * window_size - len(word_context)
                if padd_num > 0:
                    word_contexts.append(word_context + [self.padding_idx for i in range(padd_num)])
                else:
                    word_contexts.append(word_context)
        
        self.word_indeces = torch.tensor(word_indeces,dtype=torch.long)
        self.context_indeces = torch.tensor(word_contexts,dtype=torch.long)
        
    def __getitem__(self,idx):
        return self.word_indeces[idx], self.context_indeces[idx,:]
    
    def __len__(self):
        return len(self.word_indeces)
        
        
        
        
                
class SkipGramTrainer(object):
    
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
        dataset = SkipGramSamples(corpus, window_size=window_size, cutoff_freq=cutoff_freq)
        samples_idx = torch.randperm(len(dataset))
        my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))
        self.batch_len = len(self.dataloader)
        self.vocab_size = len(dataset.vocabulary)    
        
        print('SkipGram trainer created:')
        print('Window size: {}'.format(window_size))
        print('Number of samples: {}'.format(len(dataset)))
        print('Vocabulary Size: {}'.format(self.vocab_size))
        print('Number of batches: {}'.format(self.batch_len))
        print('Number of samples per batch: {}'.format(batch_size))
        print()

        
    def InitModel(self, state_dict=None, device='cpu', paralelize=False, **kwargs):
        
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
        
        self.model = SkipGramModel(self.vocab_size, self.embedding_dim)
        
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
        
        try:
            for e in range(epochs):
                for t, (x,y) in enumerate(self.dataloader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    
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
        
        
class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        
    def forward(self,x):
        return self.out(self.emb(x))
    
    def loss(self,scores,target):
        lf = nn.CrossEntropyLoss(ignore_index=self.vocab_size,reduction='sum')
        scores = scores.view(-1,self.vocab_size,1).repeat(1,1,target.size(1))
        return lf(scores,target)

    
import re
    
def GetTrainCorpus(file):
    with open(file, 'rb') as f:
        lines = f.readlines()
        corpus = [['<s>'] + re.split(r'[\t \n]',l.decode('iso-8859-1'))[2:-1] + ['</s>'] for l in lines]
    return corpus
    
    
def GetEmbeddings(file, trainer):
    with open(file, 'rb') as f:
        lines = f.readlines()
        corpus = [['<s>'] + re.split(r'[\t \n]',l.decode('iso-8859-1'))[2:-1] + ['</s>'] for l in lines]
    test_vocab = Vocabulary.from_corpus(corpus,cutoff_freq=0)
    train_vocab = trainer.dataloader.dataset.vocabulary
    embedding_dict = {}
    embedding_dim = len(trainer.model.out.weight.data[0,:])
    for tk in test_vocab:
        try:
            idx = train_vocab[tk]
            embedding_dict[tk] = (trainer.model.emb(torch.tensor(idx)), trainer.model.out.weight.data[idx,:])
        except:
            embedding_dict[tk] = (torch.randn(embedding_dim), torch.randn(embedding_dim))
    
    return embedding_dict

    
    
    
    
    