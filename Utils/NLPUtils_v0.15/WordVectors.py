import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn as nn



class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self):

        self._token_to_idx = {}
        self._idx_to_token = {}
        self._idx_to_freq = {}

    def add_token(self, token):
        
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
            self._idx_to_freq[index] += 1
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            self._idx_to_freq[index] = 1
        return index
    
    def index_to_token(self, index):
        
        if not isinstance(index, list):
            if not isinstance(index, int):
                raise NameError("'index' must be an integer or list of integers")
            if index not in self._idx_to_token:
                raise KeyError('the index {} exeeds the Vocabulary lenght'.format(index))
            return self._idx_to_token[index]
        
        tokens = []
        for idx in index:
            if not isinstance(idx, int):
                raise NameError("{} is not an integer".format(idx))
            if idx not in self._idx_to_token:
                raise KeyError('the index {} exeeds the Vocabulary lenght'.format(idx))
            tokens.append(self._idx_to_token[idx])
        return tokens

    def token_to_index(self, token):
        
        if not isinstance(token, list):
            if not isinstance(token, str):
                raise NameError("'token' must be a string or list of strings")
            if token not in self._token_to_idx:
                raise KeyError('the token {} is not in the Vocabulary'.format(token))
            return self._token_to_idx[token]
        
        indeces = []
        for tk in token:
            if not isinstance(tk, str):
                raise NameError("'token' must be a string or list of strings")
            if tk not in self._token_to_idx:
                raise KeyError('the token {} is not in the Vocabulary'.format(tk))
            indeces.append(self._token_to_idx[tk])
        return indeces
    
    def get_freq(self, tk_or_idx):
        
        if isinstance(tk_or_idx, int):
            if tk_or_idx not in self._idx_to_token:
                raise KeyError('the index {} exeeds the Vocabulary lenght'.format(tk_or_idx))
            freq = self._idx_to_freq[tk_or_idx]
        elif isinstance(tk_or_idx, str):
            if tk_or_idx not in self._token_to_idx:
                freq = 0
            else:
                freq = self._idx_to_freq[self._token_to_idx[tk_or_idx]]
        else:
            raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
        return freq

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)
    
    def __getitem__(self,idx):
        return self.index_to_token(idx)


class Word2VecSamples(Dataset):
    
    no_token = '<NT>'
    
    def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
        # Obtengo el vocabulario a partir del corpus ya tokenizado:
        self.corpus = corpus
        self.vocabulary = Vocabulary()
        for doc in corpus:
            for token in doc:
                self.vocabulary.add_token(token)
                
        # Obtengo el contexto a partir del corpus: token if self.vocabulary.get_freq(token) >= cutoff_freq else self.no_token \
        self.window_size = window_size
        self.data = pd.DataFrame({'word': [token for doc in corpus for token in doc],
                                  'context': [[self.no_token for j in range(i-window_size, max(0,i-window_size))] + \
                                              doc[max(0,i-window_size):i] + \
                                              doc[i+1:min(i+window_size+1, len(doc))] + \
                                              [self.no_token for j in range(min(i+window_size+1, len(doc)),i+window_size+1)] \
                                              for doc in corpus for i in range(len(doc))]
                                 })
        self.padding_idx = len(self.vocabulary)
        
    def __getitem__(self,idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        
        word_vector = torch.tensor(self.vocabulary.token_to_index(self.data['word'].iloc[idx]), dtype=torch.long)
        context_vector = torch.zeros(2 * self.window_size, dtype=torch.long)
        for i, token in enumerate(self.data['context'].iloc[idx]):
            if token == self.no_token:
                context_vector[i] = self.padding_idx
            else:
                context_vector[i] = self.vocabulary.token_to_index(token)
            
        return word_vector, context_vector        
    
    def __len__(self):
        return len(self.data)
    
    
    
    
class CBOWModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel,self).__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self,x):
        embedding = self.emb(x).mean(dim=1)
        return self.out(embedding)
    
    def loss(self,scores,target):
        lf = nn.CrossEntropyLoss()
        return lf(scores,target)
        
        
class SkipGramModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel,self).__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        
    def forward(self,x):
        return self.out(self.emb(x))
    
    def loss(self,scores,target):
        lf = nn.CrossEntropyLoss(ignore_index=self.vocab_size)
        if target.size() != torch.Size([2]):
            context_size = target.size(1)
            scores = scores.view(-1,self.vocab_size,1).repeat(1,1,context_size)
        return lf(scores,target)