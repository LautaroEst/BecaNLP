import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn as nn
import numpy as np



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
    
#     def _read_corpus(self,corpus,cutoff_freq=0):
        
#         temp_vocabulary = Vocabulary()
#         for doc in corpus:
#             for token in doc:
#                 temp_vocabulary.add_token(token)
        
#         if cutoff_freq == 0:
#             return temp_vocabulary
        
#         vocabulary = Vocabulary()
#         i = 0
#         for idx, freq in temp_vocabulary._idx_to_freq.items():
#             if freq >= cutoff_freq:
#                 i += 1
#                 vocabulary.add_token(temp_vocabulary.index_to_token(idx))
#                 vocabulary._idx_to_freq[i] = freq
        
#         return vocabulary
    
    
#     def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
#         # Obtengo el vocabulario a partir del corpus ya tokenizado:
#         self.vocabulary = self._read_corpus(corpus,cutoff_freq)
        
#         # Obtengo el contexto a partir del corpus:
#         self.padding_idx = len(self.vocabulary)
#         self.window_size = window_size
#         total_words = sum([len(doc) for doc in corpus])
#         self.word_indeces = np.zeros(total_words,dtype=np.int)
#         self.context_indeces = np.zeros((total_words,window_size*2),dtype=np.int)
        
#         n = 0
#         for doc in corpus:
#             for t, token in enumerate(doc):
#                 self.word_indeces[n] = self.vocabulary._token_to_idx.get(token,self.padding_idx)
#                 self.context_indeces[n,:] = np.array([
#                 [self.padding_idx for j in range(t-window_size, max(0,t-window_size))] + \
#                 [self.vocabulary._token_to_idx.get(tk,self.padding_idx) for tk in doc[max(0,t-window_size):t]] + \
#                 [self.vocabulary._token_to_idx.get(tk,self.padding_idx) for tk in doc[t+1:min(t+window_size+1, len(doc))]] + \
#                 [self.padding_idx for j in range(min(t+window_size+1, len(doc)),t+window_size+1)]])
#                 n += 1
        
#         self.word_indeces = torch.from_numpy(self.word_indeces)
#         self.context_indeces = torch.from_numpy(self.context_indeces)


    def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
        # Obtengo el vocabulario a partir del corpus ya tokenizado:
        self.vocabulary = Vocabulary()
        for doc in corpus:
            for token in doc:
                self.vocabulary.add_token(token)
        
        # Obtengo el contexto a partir del corpus:
        self.padding_idx = len(self.vocabulary)
        self.window_size = window_size
        
        word_indeces = []
        word_contexts = []
        
        for doc in corpus:
            for t, token in enumerate(doc):
                if self.vocabulary.get_freq(token) > cutoff_freq:
                    complete_context = [self.no_token for j in range(t-window_size, max(0,t-window_size))] + \
                    [tk for tk in doc[max(0,t-window_size):t]] + \
                    [tk for tk in doc[t+1:min(t+window_size+1, len(doc))]] + \
                    [self.no_token for j in range(min(t+window_size+1, len(doc)),t+window_size+1)]
                    mask = [tk != self.no_token and self.vocabulary.get_freq(tk) > cutoff_freq \
                            for tk in complete_context]
                    if any(mask):
                        word_indeces.append(self.vocabulary.token_to_index(token))
                        word_contexts.append([self.vocabulary.token_to_index(complete_context[i]) \
                                              if mask[i] else self.padding_idx for i in range(2*window_size)])
                
        self.word_indeces = torch.tensor(word_indeces,dtype=torch.long)
        self.context_indeces = torch.tensor(word_contexts,dtype=torch.long)
        

        
    def __getitem__(self,idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        return self.word_indeces[idx], self.context_indeces[idx,:]
    
    def __len__(self):
        return len(self.word_indeces)
    
    
    
    
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