import nltk
import numpy as np
import torch
import torch.nn as nn

special_tokens = {
    'unk_tk': '<unk>',
    'start_tk': '<s>',
    'end_tk': '</s>'    
}


def tokenize_dataseries(ds, vocab, include_unk=True, include_startend=True, language='english'):
    """
    Función de tokenización estándar. Usa NLTK. Se asume que ya se identificaron
    las palabras que componen el vocabulario, y se le agregan a éstas los tokens
    unk, start y end.
    """
    ds = ds.str.lower().apply(nltk.tokenize.word_tokenize,args=(language, False))

    tk_to_idx = {tk: idx for idx, tk in enumerate(vocab)}
    num_tokens = len(tk_to_idx)
    if include_unk:
        tk_to_idx[special_tokens['unk_tk']] = num_tokens
        unk_idx = num_tokens
        num_tokens += 1    
    if include_startend:
        tk_to_idx[special_tokens['start_tk']] = num_tokens
        s_idx = num_tokens
        tk_to_idx[special_tokens['end_tk']] = num_tokens + 1
        bs_idx = num_tokens + 1
        num_tokens += 2

    if include_startend:
        comments = [[s_idx] + [tk_to_idx.get(tk,unk_idx) for tk in row] + [bs_idx] \
                    for row in ds]
    else:
        comments = [[tk_to_idx.get(tk,unk_idx) for tk in row] for row in ds]
    
    idx_to_tk = {idx: tk for idx, tk in enumerate(tk_to_idx)}
    return comments, idx_to_tk
    

def idx_to_bow(idx_list, idx_to_tk, include_unk, dtype=torch.long):
    """
    Convierte una lista de listas de índices en una matriz cuyas
    filas son vectores BOW con los índices. Si la lista contiene
    N listas, la matriz tiene dimensiones N x vocab_size.
    """
    num_samples = len(idx_list)
    vocab_size = len(idx_to_tk)
    unk_idx = list(idx_to_tk.values()).index(special_tokens['unk_tk'])

    if include_unk:
        x = np.zeros((num_samples,vocab_size))
        for i, sample in enumerate(idx_list):
            unique_sample, count_uniques = np.unique(sample,return_counts=True)
            x[i,unique_sample] += count_uniques
    else:
        x = np.zeros((num_samples,vocab_size-1))
        for i, sample in enumerate(idx_list):
            unique_sample, count_uniques = np.unique(sample,return_counts=True)
            mask = unique_sample != unk_idx
            x[i,unique_sample[mask]] += count_uniques[mask]
    return torch.from_numpy(x).type(dtype)

    
def idx_to_sequence(idx_list,dtype=torch.long):
    """
    Devuelve la misma lista en un tensor, y llena los faltantes
    con el valor en padding.
    """
    x = [torch.tensor(sample).view(-1,1) for sample in idx_list]    
    return x



