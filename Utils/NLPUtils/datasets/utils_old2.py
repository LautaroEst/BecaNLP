from itertools import islice, tee
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

def get_count_vectorizer(reader,**kwargs):
    """
    Función para inicializar el vectorizer cuando se hace bag of n-grams.
    Recibe un reader, que es in iterable que devuelve en cada muestra
    un texto, y los parámetros del CountVectorizer de scikit-learn
    """
    vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(reader)
    return vectorizer

def split_dev_kfolds(N,dev_size=.2,k_folds=None,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    indeces = []
    if k_folds is None:
        N_train = int(N * (1-dev_size))
        if N_train == N and dev_size != 0:
            print('Warning: dev_size too small!')
            N_train = N-1
        indeces.append((rand_idx[:N_train], rand_idx[N_train:]))
    else:
        splitted_arrays = np.array_split(rand_idx,k_folds)
        for i in range(1,k_folds+1):
            train_idx = np.hstack(splitted_arrays[:i-1] + splitted_arrays[i:])
            dev_idx = splitted_arrays[i-1]
            indeces.append((train_idx, dev_idx))

    return indeces


def ngrams(lst, n):
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break


class NgramTokenizer(object):
    def __init__(self,token_pattern,ngram_range):
        self.token_pattern = token_pattern

    def __call__(self,text):
        tokens = re.split(token_pattern,text)
        for i in range(ngram_range[0],ngram_range[1]):
            tokens.append()
        return 



class TextVectorizer(object):

    def __init__(self,token_pattern=r'\bw+\b',ngram_range=(1,1),
        vocabulary=None,max_features=None,unk_token='<UNK>'):
        
        self.tokenizer = self.NgramTokenizer(token_pattern,ngram_range)
        self.ngram_range = ngram_range
        



        self.max_features = max_features
        self.unk_token = unk_token

    def _count_vocab(self,corpus):
        
        # caso en que vocabulary == None
        tk_to_freq = Counter()
        for text in corpus:
            tk_to_freq += Counter([tk for tk in self.tokenizer(text)])






