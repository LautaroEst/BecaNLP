import torch
from torch.utils.data import Dataset
import itertools


class TextClassificationDataset(Dataset):
    
    def __init__(self, corpus_examples_idx, vocab_examples, corpus_labels_idx, vocab_labels, encoding):
        self.corpus_examples_idx = corpus_examples_idx
        self.vocab_examples = vocab_examples
        self.corpus_labels_idx = corpus_labels_idx
        self.vocab_labels = vocab_labels
        
        if encoding == 'index':
            self.set_getitem(lambda self, idx: self.corpus_examples_idx[idx])
        elif encoding == 'one-hot':
            pass
        elif encoding == 'tf-idf':
            pass
        elif encoding == 'ppmi':
            pass
        else:
            raise ValueError('Encodificación no válida')
        
    @classmethod
    def set_getitem(cls, getitem_fn):
        cls.__getitem__ = getitem_fn
    
    @classmethod
    def from_dataframe(cls,df,tokenizer=None,vocabulary=None,encoding='index'):
        if tokenizer is None:
            tokenizer = lambda x: x.split(' ')
        corpus_examples = df.iloc[:,0].apply(tokenizer).tolist()
        vocab_examples = get_vocabulary_from_list_of_list_of_tokens(corpus_examples,words=vocabulary)
        corpus_examples_idx = [[vocab_examples.get(tk,-1)[1] for tk in doc] for doc in corpus_examples]
        
        corpus_labels = df.iloc[:,1].tolist()
        vocab_labels = sorted(list(set(corpus_labels)))
        corpus_labels_idx = [vocab_labels.index(tk) for tk in corpus_labels]
        
        return cls(corpus_examples_idx, vocab_examples, corpus_labels_idx, vocab_labels, encoding)

    
class LMDatasets(Dataset):
    
    def __init__(self):
        pass

    
    
    
def get_vocabulary_from_list_of_list_of_tokens(corpus,words=None):
    if words is None:
        words = sorted(list(set(itertools.chain.from_iterable(corpus))))
    else:
        words = sorted(words)
    freqs_idx_dict = {tk: [0,idx] for idx, tk in enumerate(words)}
    for tk in itertools.chain.from_iterable(corpus):
        try:
            freqs_idx_dict[tk][0] += 1
        except KeyError:
            continue
    return freqs_idx_dict
    

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
    
    