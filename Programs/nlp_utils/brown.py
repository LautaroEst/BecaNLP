import torch

import re
import itertools
import nltk
nltk.download('brown', download_dir='/home/lestien/Documents/BecaNLP/Programs/nltk_data')
from nltk.corpus import brown


def GetBrownClassificationDataset(test_prop, preprocessing=None):

    categories = brown.categories()
    
    # Preprocesamiento del corpus:
    corpus_unpreproceced = brown.sents(categories=categories)
    if preprocessing:
        corpus = preprocessing(corpus_unpreproceced)
    else:
        corpus = corpus_unpreproceced
        
    # Definición del vocabulario:
    vocabulary = set(itertools.chain.from_iterable(corpus))
            
    # Split del corpus y definición de las muestras:
    train_samples = []
    test_samples = []

    for c, category in enumerate(categories):
        sents = brown.sents(categories=category)
        categ_len = len(sents)
        test_size = int(test_prop * categ_len)
        train_size = categ_len - val_size - test_size
        rand_idx = torch.randperm(categ_len)
        for i in rand_idx[:train_size]:
            train.append(sents[i])
            train_labels.append(c)
        for i in rand_idx[train_size:(train_size+val_size)]:
            val.append(sents[i])
            val_labels.append(c)
        for i in rand_idx[-test_size:]:
            test.append(sents[i])
            test_labels.append(c)

    train_samples, train_rand_idx = [], torch.randperm(len(train))
    val_samples, val_rand_idx = [], torch.randperm(len(val))
    test_samples, test_rand_idx = [], torch.randperm(len(test))

    for i in train_rand_idx:
        train_samples.append((train[i], train_labels[i]))
    for i in val_rand_idx:
        val_samples.append((val[i], val_labels[i]))
    for i in test_rand_idx:
        test_samples.append((test[i], test_labels[i]))
    
    train_dataset = BrownDataset(train_samples, vocabulary)
    val_dataset = BrownDataset(val_samples, vocabulary)
    test_dataset = BrownDataset(test_samples, vocabulary)


class BrownDataset(torch.utils.data.Dataset):
    
    def __init__(self, samples, vocabulary):
        
        self.categories = brown.categories()
        self.vocabulary = vocabulary
        self.word_to_index = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(self.vocabulary)}
        
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        
        context, word = self.samples[idx]
        idx_context = torch.empty(len(context), dtype=torch.long)
        idx_word = torch.tensor(self.word_to_index[word], dtype=torch.long)
        for i, w in enumerate(context):
            idx_context[i] = self.word_to_index[w]

        return idx_context, idx_word
       
