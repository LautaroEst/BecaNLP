import nltk
import numpy as np
import torch

def tokenize_dataseries(ds, vocab, unk_idx=-1):
    tk_to_idx = {tk: idx for idx, tk in enumerate(vocab)}
    ds = ds.str.lower().apply(nltk.tokenize.word_tokenize,args=('english', False))
    comments = [[tk_to_idx.get(tk,unk_idx) for tk in row] for row in ds]
    return comments
    

def idx_to_bow(idx_list, vocab_size, dtype=torch.long):
    num_samples = len(idx_list)
    x = torch.zeros(num_samples,vocab_size,dtype=dtype)
    for i, sample in enumerate(idx_list):
        for j in sample:
            if j!= vocab_size:
                x[i,j] += 1.
    return x

    
def idx_to_sequence(idx_list,padding=-1,dtype=torch.long):
    num_samples = len(idx_list)
    lengths = np.array([len(sample) for sample in idx_list])
    max_len = lenghts.max()
    x = torch.zeros(num_samples,max_len,dtype=dtype) + padding
    for i, sample in enumerate(idx_list):
        x[i,:lenghts[i]] = torch.tensor(sample)
    return x