import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn as nn
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
    
    
class Word2VecSamples(Dataset):
    
    unk_token = '<UNK>'
    
    def samples_generator(self, doc):
        for t, token in enumerate(doc):
            if token in self.vocabulary:
                len_doc = len(doc)
                cond1 = max(-1,t-self.window_size) == -1
                cond2 = min(t+self.window_size, len_doc) == len_doc
                if cond1 and cond2:
                    context = itertools.chain(doc[:t],doc[t+1:])
                if cond1 and not cond2:
                    context = itertools.chain(doc[:t],doc[t+1:t+self.window_size+1])
                if cond2 and not cond1:
                    context = itertools.chain(doc[t-self.window_size:t],doc[t+1:])
                if not cond1 and not cond2:
                    context = itertools.chain(doc[t-self.window_size:t],doc[t+1:t+self.window_size+1])

                context_list = [self.vocabulary.token_to_index(tk) for tk in context if tk in self.vocabulary]
                if len(context_list) != 0:
                    yield (self.vocabulary.token_to_index(token), context_list)
    

    def __init__(self, corpus, window_size=2, cutoff_freq=0):
        
        # Obtengo el vocabulario a partir del corpus ya tokenizado:
        self.vocabulary = Vocabulary.from_corpus(corpus,cutoff_freq=cutoff_freq)
    
        # Obtengo el contexto a partir del corpus:
        self.padding_idx = len(self.vocabulary)
        self.window_size = window_size
        
        word_indeces = []
        word_contexts = []
        for doc in corpus:
            gen = self.samples_generator(doc)
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

    
    
class CBOWModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel,self).__init__()
        self.emb = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self,x):
        embedding = self.emb(x).mean(dim=1)
        return self.out(embedding)
    
    def loss(self,scores,target):
        lf = nn.CrossEntropyLoss(reduction='sum')
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
        lf = nn.CrossEntropyLoss(ignore_index=self.vocab_size,reduction='sum')
        scores = scores.view(-1,self.vocab_size,1).repeat(1,1,target.size(1))
        return lf(scores,target)
    
    
    
    