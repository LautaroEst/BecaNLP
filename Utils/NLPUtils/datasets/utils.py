import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import time

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

class NgramTextVectorizer(CountVectorizer):
    
    def __init__(self,token_pattern=r'\b\w+\b',vocabulary=None,unk_token=None,
         min_freq=1,max_freq=np.inf,ngram_range=(1,1),max_features=None):
        
        super().__init__(lowercase=False,token_pattern=token_pattern,
            vocabulary=None,ngram_range=ngram_range)

        self.unk_token = unk_token
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.open_vocab = True if vocabulary is None else False
        self.true_vocab = vocabulary
        self.true_max_features = max_features
        

    def get_columns_to_keep(self,X,min_freq,max_freq):
        if min_freq <= 0 and max_freq == np.inf:
            return np.arange(X.shape[1])

        sums = np.asarray(X.sum(axis=0)).reshape(-1)
        keep_items = np.where(np.logical_and(sums >= min_freq,sums <= max_freq))[0]
        return keep_items

    def remove_items(self,X,items_to_keep):

        if len(items_to_keep) == 0:
            raise RuntimeError('Vocabulary sin tokens')

        vocab = self.vocabulary_
        vocab_len = len(vocab)
        
        if self.unk_token is not None:
            remove_tokens = np.ones(vocab_len,dtype=np.bool)
            remove_tokens[items_to_keep] = False
            X = X.tolil()
            X[:,-1] = X[:,remove_tokens].sum(axis=1)
            X = X.tocsr()
        
        X = X[:,items_to_keep]
        
        sorted_idx = np.argsort(list(vocab.values()))
        tokens = list(vocab.keys())
        sorted_tokens = [tokens[i] for i in sorted_idx]
        self.vocabulary_ = {sorted_tokens[i]:idx for idx, i in enumerate(items_to_keep)}
        return X


    def fit_transform(self,corpus):
        X = super().fit_transform(corpus)
        if self.open_vocab:
            # tengo que recortar por frecuencia y/o por max_features
            keep_items = self.get_columns_to_keep(X,self.min_freq,self.max_freq)
            

        else:
            # tengo que sacar las palabras que no pertenecen al vocabulario
            # y agregar las que sí pertenecen
            vocab = self.vocabulary_
            true_vocab = self.true_vocab
            keep_items = [idx for ngram, idx in vocab.items() if all([tk in true_vocab for tk in ngram.split(' ')])]

            for tk in true_vocab:
                try:
                    _ = vocab[tk]
                except KeyError:
                    len_vocab = len(vocab)
                    keep_items.append(len_vocab)
                    vocab[tk] = len_vocab
            X.resize(X.shape[0],X.shape[1]+len(keep_items))
            keep_items = np.array(keep_items)

        if self.true_max_features is not None:
            if self.true_max_features < len(keep_items):
                sums = np.asarray(X[:,keep_items].sum(axis=0)).reshape(-1)
                sums_idx = np.argsort(sums)[::-1][:self.true_max_features]
                most_frequents = keep_items[sums_idx]
                keep_items = most_frequents

        # si elijo no ignorar las palabras desconocidas, agrego
        # el token al vocabulario y una columna a la matriz
        unk_token = self.unk_token
        if unk_token is not None:
            vocab_len = len(self.vocabulary_)
            self.vocabulary_[unk_token] = vocab_len
            keep_items = np.hstack((keep_items,np.array([vocab_len])))
            X.resize(X.shape[0],X.shape[1]+1)

        X = self.remove_items(X,keep_items)
        return X

    
    def fit(self,corpus):
        self.fit_transform(corpus)
        return self


    def transform(self,corpus):
        unk_token = self.unk_token
        vocab = self.vocabulary_
        
        indptr = [0]
        j_indices = []
        values = []
        
        tokenizer = self.build_analyzer()
        if unk_token is None:
            for doc in corpus:
                feature_counter = {}
                for feature in tokenizer(doc):
                    try:
                        feature_idx = vocab[feature]
                        if feature_idx not in feature_counter:
                            feature_counter[feature_idx] = 1
                        else:
                            feature_counter[feature_idx] += 1
                    except KeyError:
                        pass

                j_indices.extend(feature_counter.keys())
                values.extend(feature_counter.values())
                indptr.append(len(j_indices))
                
        else:
            for doc in corpus:
                feature_counter = {}
                for feature in tokenizer(doc):
                    try:
                        feature_idx = vocab[feature]
                    except KeyError:
                        feature_idx = vocab[unk_token]

                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1

                j_indices.extend(feature_counter.keys())
                values.extend(feature_counter.values())
                indptr.append(len(j_indices))
            
        j_indices = np.asarray(j_indices, dtype=self.dtype)
        indptr = np.asarray(indptr, dtype=self.dtype)
        values = np.asarray(values, dtype=self.dtype)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocab)),
                          dtype=self.dtype)
        X.sort_indices()
            
        return X
        
    
        
