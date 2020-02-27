import torch
import re
import numpy as np
from .utils import *

def GetARPAFile(trainer, train_corpus, output_file):
    
    test_text = read_text_file('./promptsl40_test_cleaned')
    test_corpus = split(test_text, r'[ \n]')

    # Probabilidad de los unigramas:
    test_vocab = Vocabulary.from_list_corpus([test_corpus],cutoff_freq=1)
    train_vocab = train_corpus.vocabulary
    train_vocab_size = len(train_vocab)
    unigram_probs = {}
    uniform_prob = max(-99.,-float(np.log(len(train_vocab))))
    for tk in test_vocab:
        idx = train_vocab[tk]
        if idx == train_vocab_size:
            unigram_probs[tk] = uniform_prob
            continue
        out = trainer.model.out.weight.data[idx,:].to(device=trainer.device)
        unigram_scores = trainer.model.out(out)
        unigram_probs[tk] = min(max(-99,(unigram_scores - torch.logsumexp(unigram_scores, dim=0))[idx].item()),0.)
    
    # Probabilidad de los bigramas:
    len_test_corpus = len(test_corpus)
    test_bigrams = sorted(list(set(['{} {}'.format(test_corpus[t-1], test_corpus[t]) for t in range(1,len_test_corpus)])))
    bigram_probs = {}
    for bigram in test_bigrams:
        w1, w2 = bigram.split(' ')
        idx1, idx2 = train_vocab[w1], train_vocab[w2]
        if idx1 == train_vocab_size or idx2 == train_vocab_size:
            bigram_probs[bigram] = uniform_prob
            continue
        x = torch.tensor(idx1).to(device=trainer.device)
        bigram_scores = trainer.model.out(trainer.model.emb(x))
        bigram_probs[bigram] = min(max(-99.,(bigram_scores - torch.logsumexp(bigram_scores, dim=0))[idx2].item()),0.)
        
    # Creo el archivo en formato ARPA:
    new_lm_file = ['\n', 
                   '\\data\\\n', 
                   'ngram 1={}\n'.format(len(test_vocab)), 
                   'ngram 2={}\n'.format(len(test_bigrams)), 
                   '\n', 
                   '\\1-grams:\n']
    new_lm_file += ['{:.6f}\t{}\t0\n'.format(p,unigram) for unigram, p in unigram_probs.items()]
    new_lm_file += ['\n'] + ['\\2-grams:\n'] 
    new_lm_file += ['{:.6f}\t{}\n'.format(p,bigram) for bigram, p in bigram_probs.items()]
    new_lm_file += ['\n', '\\end\\\n']

    with open(output_file, 'wb') as file:
        file.write((''.join(new_lm_file)).encode('iso-8859-1'))
    
    return unigram_probs, bigram_probs
