import torch
import pandas as pd


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
    
    
class Vectorizer(object):
    
    def __init__(self, data, cutoff=25):
        
        self.vocabulary = Vocabulary()
        for sample in data:
            for token in sample:
                self.vocabulary.add_token(token)
        self.num_tokens = len(self.vocabulary)
        self.cutoff = cutoff
    
    def vectorize(self,sample):
        one_hot = torch.zeros(self.num_tokens,dtype=torch.float)
        for token in sample:
            if self.vocabulary.get_freq(token) >= self.cutoff:
                one_hot[self.vocabulary.token_to_index(token)] = 1
        return one_hot    