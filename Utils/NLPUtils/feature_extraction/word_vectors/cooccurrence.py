from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
from ..count import count_bag_of_ngrams


def _filter_by_token(cooccurrences_dict, tokens=None, negative=True):
    if tokens is None:
        return cooccurrences_dict
    
    if negative:
        new_dict = {key:val for key, val in cooccurrences_dict.items() if key[0] not in tokens and key[1] not in tokens}
    else:
        new_dict = {key:val for key, val in cooccurrences_dict.items() if key[0] in tokens and key[1] in tokens}
    return new_dict


def word_by_word_cooccurrence(corpus, window=None, left_n=2, right_n=2, 
    tokenizer=None, min_count=0., max_count=None, max_words=None):
    """
    Devuelve una matriz de coocurrencias entre palabras y una ventana de palabras cercanas.
    Las filas son las palabras y las columnas, los features. Es decir, es una matriz cuadrada.
    """

    if tokenizer is None:
        tokenizer = lambda x: x

    cooccurrences_dict = defaultdict(float)
    full_vocab = defaultdict()
    full_vocab.default_factory = full_vocab.__len__

    unk_idx = -1
    if window is None:
        window = [1. for i in range(left_n+right_n+1)]
    else:
        if len(window) != left_n + right_n + 1:
            raise RuntimeError('El tamaño de la ventana tiene que coincidir con el tamaño del contexto')
    for doc in tqdm(corpus):
        indices = [full_vocab[tk] for tk in tokenizer(doc)]
        for i in range(left_n):
            indices.insert(0,unk_idx)
        for i in range(right_n):
            indices.append(unk_idx)
        for i, idx in enumerate(indices):
            context = indices[i-left_n:i+right_n+1]
            for j, c in zip(window,context):
                cooccurrences_dict[(idx, c)] += j

    cooccurrences_dict = _filter_by_token(dict(cooccurrences_dict), [unk_idx], negative=True)
    full_vocab = dict(full_vocab)
    i, j = zip(*cooccurrences_dict.keys())
    data = list(cooccurrences_dict.values())
    vocab_len = len(full_vocab)
    X = coo_matrix((data, (i,j)),shape=(vocab_len,vocab_len)).tocsr()

    # Limito por frecuencia o por tope máximo de palabras
    freqs = np.array(X.diagonal()).reshape(-1)
    if min_count <= 0 and max_count is None:
        sorted_idx = np.argsort(freqs)[::-1]
        if max_words is not None:
            sorted_idx = sorted_idx[:max_words]

    else:
        if max_count is None:
            max_count = np.inf

        mask = np.logical_and(freqs <= max_count, freqs >= min_count)
        X = X[mask,:]
        X = X[:,mask]
        sorted_idx = np.argsort(freqs[mask])[::-1]
        if max_words is not None:
            sorted_idx = sorted_idx[:max_words]

    sorted_keys = np.array(list(full_vocab.keys()))[sorted_idx]
    vocab = {tk:idx for idx,tk in enumerate(sorted_keys)}
    X = X[sorted_idx,:]
    X = X[:,sorted_idx]
    return X, full_vocab


def word_by_document_cooccurrence(corpus, tokenizer=None, 
    min_count=0., max_count=None, max_words=None):
    """
    Devuelve la matriz de coocurrencias entre palabras y documentos. Cada fila es una palabra
    y cada columna, un documento distinto.
    """
    X, vocab = count_bag_of_ngrams(corpus, ngram_range=(1,1), tokenizer=tokenizer)
    return _reduce_by_freq(X.T.tocsr(), vocab, min_count, max_count, max_words)


def word_by_category_cooccurrence(corpus, labels, tokenizer=None,
    min_count=0., max_count=None, max_words=None):
    """
    Devuelve la matriz de coocurrencias entre palabras y la categoría a la que pertence 
    el documento. Es decir, las filas de la matriz son las palabras y las columnas son
    todas las categorías posibles, y todas las entradas de la matriz contienen la cuenta
    de cuántas veces apareció la palabra en un documento de cada categoría.
    """
    categories = sorted(set(labels)) # Se asume que los labels son 0, 1, ..., len(categories)
    cooccurrences_dict = defaultdict(float)
    full_vocab = defaultdict()
    full_vocab.default_factory = full_vocab.__len__

    if tokenizer is None:
        tokenizer = lambda x: x

    for doc, label in zip(corpus, labels):
        for tk in tokenizer(doc):
            cooccurrences_dict[(full_vocab[tk],label)] += 1.

    full_vocab = dict(full_vocab)
    i, j = zip(*cooccurrences_dict.keys())
    data = list(cooccurrences_dict.values())
    X = coo_matrix((data, (i,j)),shape=(len(full_vocab),len(categories)))
    return _reduce_by_freq(X.tocsr(), full_vocab, min_count, max_count, max_words)


def _reduce_by_freq(X, vocab, min_count=0., max_count=None, max_words=None):

    freqs = np.array(X.sum(axis=1)).reshape(-1)

    if min_count <= 0 and max_count is None:
        sorted_idx = np.argsort(freqs)[::-1]
        if max_words is not None:
            sorted_idx = sorted_idx[:max_words]

    else:
        if max_count is None:
            max_count = np.inf

        mask = np.logical_and(freqs <= max_count, freqs >= min_count)
        X = X[mask,:]
        sorted_idx = np.argsort(freqs[mask])[::-1]
        if max_words is not None:
            sorted_idx = sorted_idx[:max_words]


    sorted_keys = np.array(list(vocab.keys()))[sorted_idx]
    vocab = {tk:idx for idx,tk in enumerate(sorted_keys)}
    X = X[sorted_idx,:]
    return X, vocab
