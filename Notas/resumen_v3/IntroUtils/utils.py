import re
import itertools
import pandas as pd

class Vocabulary(object):
    
    unk_token = '<UNK>'

    def __init__(self, tokens_dict=None):
        
        self._idx_to_tk = {} if tokens_dict is None else tokens_dict
        self._tk_to_idx = {} if tokens_dict is None else {tk: idx for idx, tk in tokens_dict.items()}
        self.max_idx = len(self)
        
    @classmethod
    def from_corpus(cls, corpus, cutoff_freq=1):
        corpus_words = sorted(list(set([tk for doc in corpus for tk in doc])))
        freqs_dict = {word: 0 for word in corpus_words}
        for tk in itertools.chain.from_iterable(corpus):
            freqs_dict[tk] += 1
        new_corpus_words = {idx: tk for idx, tk in enumerate(corpus_words) if freqs_dict[tk] >= cutoff_freq}
        return cls(new_corpus_words)
    
    @classmethod
    def from_string(cls, corpus, delimiter=' ', cutoff_freq=1):
        corpus = [corpus.split(delimiter)]
        return cls.from_corpus(corpus, cutoff_freq)
    
    @classmethod
    def from_wordlist(cls, corpus):
        tokens_dict = {idx: tk for idx, tk in enumerate(corpus)}
        return cls(tokens_dict)
    
    @classmethod
    def from_dataseries(cls, ds, delimiter=' ', cutoff_freq=1):
        corpus = [doc for doc in ds.str.split(delimiter)]
        return cls.from_corpus(corpus, cutoff_freq)

    def index_to_token(self, index):
        return self._idx_to_tk[index]

    def token_to_index(self, token):
        if token not in self._tk_to_idx.keys():
            return self.max_idx
        return self._tk_to_idx[token]

    def __repr__(self):
        return "<Vocabulary(size={})>".format(len(self))
    
    def __str__(self):
        text = ''
        for i, tk in enumerate(self):
            text += tk + '\n'
            if i > 10:
                text += '...'
                break
        return text

    def __len__(self):
        return len(self._idx_to_tk)
    
    def __getitem__(self,tk_or_idx):
        if isinstance(tk_or_idx, int):
            return self.index_to_token(tk_or_idx)
        if isinstance(tk_or_idx, str):
            return self.token_to_index(tk_or_idx)
        raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
    def __iter__(self):
        return (tk for tk in self._idx_to_tk.values())

    def __contains__(self,key):
        return key in self._idx_to_tk.values()
    
    def __add__(self,vocab):
        new_corpus_words = sorted(list(set(vocab._idx_to_tk.values()).union(set(self._idx_to_tk.values()))))
        new_idx_to_tk = {idx: tk for idx, tk in enumerate(new_corpus_words)}
        return Vocabulary(new_idx_to_tk)
    

def read_binary_file(filename, decode='utf-8'):
    with open(filename, 'rb') as file:
        text = file.read().decode(decode)
    return text

def read_text_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def save_to_text_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

def save_to_binary_file(text, filename, encode='utf-8'):
    with open(filename, 'wb') as file:
        file.write(text.encode(encode))

def replace(text, pattern, repl):
    return re.sub(pattern, repl, text)

def insert_between(text, pat1, pat2, insert_expr):
    return re.sub(r'({})({})'.format(pat1,pat2), r'\g<1>{}\g<2>'.format(insert_expr), text)

def insert_right(text,pattern, insert_expr):
    return insert_between(text, pattern, '', insert_expr)

def insert_left(text, pattern, insert_expr):
    return insert_between(text, '', pattern, insert_expr)

def remove(text,pattern):
    return re.sub(pattern, '', text)

def split(text, delimiter):
    return re.split(delimiter,text)