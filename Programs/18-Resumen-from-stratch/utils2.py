import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
import numpy as np
import itertools


class Vocabulary(object):
    
    unk_token = '<UNK>'

    def __init__(self, tokens_dict=None, frequencies_dict=None):
        
        self._idx_to_tk = {} if tokens_dict is None else tokens_dict
        self._idx_to_freq = {} if frequencies_dict is None else frequencies_dict
        self.max_idx = len(self)
        
    @classmethod
    def from_list_corpus(cls, corpus, cutoff_freq=1):
        corpus_words = sorted(list(set([tk for doc in corpus for tk in doc])))
        freqs_dict = {word: 0 for word in corpus_words}
        for tk in itertools.chain.from_iterable(corpus):
            freqs_dict[tk] += 1
        new_corpus_words = {idx: tk for idx, tk in enumerate(corpus_words) if freqs_dict[tk] >= cutoff_freq}
        new_freqs_dict = {idx: freqs_dict[tk] for idx, tk in enumerate(corpus_words) if freqs_dict[tk] >= cutoff_freq}
        return cls(new_corpus_words, new_freqs_dict)

    def index_to_token(self, index):
        if index == self.max_idx:
            return unk_token
        else:
            return self._idx_to_tk[index]

    def token_to_index(self, token):
        if token not in self._idx_to_tk.values():
            return self.max_idx
        for i, tk in self._idx_to_tk.items():
            if tk == token:
                return i

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
        return (tk for tk in self.idx_to_tk.values())

    def __contains__(self,key):
        return key in self._tk_to_idx
    
    def __add__(self,vocab):
        new_corpus_words = sorted(list(set(vocab._idx_to_tk.values()) + set(self._idx_to_tk.values())))
        new_idx_to_tk = {idx: tk for idx, tk in enumerate(new_corpus_words)}
        new_idx_to_freq = {idx: self._idx_to_freq[self.token_to_index(tk)] + vocab._idx_to_freq[vocab.token_to_index(tk)] \
                           for idx, tk in enumerate(new_corpus_words)}
        return Vocabulary(new_idx_to_tk, new_idx_to_freq)
    

class TextCleaner(object):
    
    def __init__(self, text):
        self.text = text
    
    @classmethod
    def from_binary_file(cls, filename, decode='utf-8'):
        with open(filename, 'rb') as file:
            text = file.read().decode(decode)
        return cls(text)
    
    @classmethod
    def from_text_file(cls, filename):
        with open(filename, 'r') as file:
            text = file.read()
        return cls(text)
    
    @classmethod
    def from_string(cls, text):
        return cls(text)
    
    def to_text_file(self, filename):
        with open(filename, 'w') as file:
            file.write(text)
        
    def to_binary_file(self, filename, encode='utf-8'):
        with open(filename, 'w') as file:
            file.write(text.encode(encode))

    @staticmethod
    def replace(text, pattern, repl):
        return re.sub(pattern, repl, text)
    
    @staticmethod
    def insert_between(text, pat1, pat2, insert_expr):
        return re.sub(r'({})({})'.format(pat1,pat2), r'\g<1>{}\g<2>'.format(insert_expr), text)
    
    @staticmethod
    def insert_right(text,pattern, insert_expr):
        return self.insert_between(text, pattern, '', insert_expr)
    
    @staticmethod
    def insert_left(text, pattern, insert_expr):
        return self.insert_between(text, '', pattern, insert_expr)

    @staticmethod
    def remove(text,pattern):
        return re.sub(pattern, '', text)
    
    @staticmethod
    def split(text, pattern, delimiter):
        text_list = re.split(pattern,delimiter,text)
        return Corpus.from_lists([text_list], cutoff_freq=1)
    
    
class Corpus(object):
    
    def __init__(self, data, cutoff_freq=1):
        self.vocabulary = Vocabulary.from_list_corpus(data, cutoff_freq=cutoff_freq)
        self.docs_num = len(data)
        self.tokens_num = sum([len(doc) for doc in data])
        self.data = [[self.vocabulary.token_to_index(tk) for tk in doc] for doc in data]
        
        self.max_idx = self.tokens_num
    
    @classmethod
    def from_binary_files(cls, filenames, decode='utf-8', delimiter_pattern=' ', cutoff_freq=0):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'rb') as file:
                    texts_list.append(file.read().decode(decode))
        elif isinstance(filenames, str):
            with open(filenames, 'rb') as file:
                texts_list.append(file.read().decode(decode))
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_text_files(cls, filenames, delimiter_pattern=' ', cutoff_freq=1):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'r') as file:
                    texts_list.append(file.read())
        elif isinstance(filenames, str):
            with open(filenames, 'r') as file:
                texts_list.append(file.read())
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_strings(cls, texts, delimiter_pattern=' ', cutoff_freq=1):
        if isinstance(texts, list):
            texts_list = texts
        elif isinstance(filenames, str):
            texts_list = [texts]
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_lists(cls, texts_list, cutoff_freq=1):
        return cls(texts_list, cutoff_freq=cutoff_freq)
    
    def __repr__(self):
        return "Corpus object\nNumber of docs = {}\nNumber of tokens = {}".format(self.docs_num, self.tokens_num)
    
    def __str__(self):
        printed_text = ''
        num_print_docs = min(self.docs_num,5)
        for i in range(num_print_docs):
            doc = self.data[i]
            if len(doc) <= 5:
                printed_text += repr([self.vocabulary.index_to_token(idx) for idx in doc]) 
            else:
                printed_text += repr([self.vocabulary.index_to_token(idx) for idx in doc[:4]])[:-1] + ', ...]'
            if i < num_print_docs:
                printed_text += '\n'
        if num_print_docs != self.docs_num:
            printed_text += '...'
        return printed_text

    def __len__(self):
        return self.tokens_num
    
    def __getitem__(self,tk_or_idx):
        if isinstance(tk_or_idx, int):
            return self.data[tk_or_idx]
        if isinstance(tk_or_idx, str):
            return [i for doc in self.data for i, tk in enumerate(doc)]
        raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
    def __iter__(self):
        return (self.vocabulary.index_to_token(idx) for doc in self.data for idx in doc)
    
    def __contains__(self,key):
        return key in self.vocabulary
                
    def append(self,corpus):
        if isinstance(corpus,Corpus):
            self.vocabulary = self.vocabulary + corpus.vocabulary
            self.docs_num += corpus.docs_num
            self.tokens_num += corpus.tokens_num
            for doc in corpus.data:
                new_doc = [self.vocabulary.token_to_index(corpus.vocabulary.index_to_token(idx)) for idx in doc]
                self.data.append(new_doc)
            self.max_idx = self.tokens_num
        else:
            raise TypeError('Sólo se puede anexar un corpus de tipo Corpus')
    
    
    
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
        
        try:
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
        
        

    


class Word2VecTrainer(Trainer):
    
    def __init__(self, model, corpus, window_size, embedding_dim, batch_size, device):
        
        vocab_size = len(corpus.vocabulary)
        
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
        lf = nn.CrossEntropyLoss(reduction='mean',ignore_idx=4)
        return lf(scores,target)
        
        
    class SkipGramModel(nn.Module):
        
        def __init__(self,vocab_size,embedding_dim):
            super().__init__()
            self.emb = nn.Embedding(vocab_size+1,embedding_dim)
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
            self.emb = nn.Embedding(vocab_size+1,embedding_dim)
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
        
        
        
    