import torch
import sys
sys.path.insert(1, '../')
from nlp_utils import *


# CÓDIGO UTILIZADO PARA OBTENER EL DATASET DE CLASIFICACIÓN A PARTIR DEL CORPUS BROWN:

test_prop = .2

train_samples = []
test_samples = []

categories = brown.categories()
for c, category in enumerate(categories):
    sents = brown.sents(categories=category)
    categ_len = len(sents)
    test_size = int(test_prop * categ_len)
    train_size = categ_len - test_size
    rand_idx = torch.randperm(categ_len)
    for i in rand_idx[:train_size]:
        train_samples.append((sents[i], '<BEGINLABEL>' + category + '<ENDLABEL>'))
    for i in rand_idx[train_size:]:
        test_samples.append((sents[i], '<BEGINLABEL>' + category + '<ENDLABEL>'))
    
train_file = open('train.txt', 'w+')
test_file = open('test.txt', 'w+')

for sample in train_samples:
    text, label = sample
    for i in range(len(text)-1):
        train_file.write(text[i])
        train_file.write('<TS>')
    train_file.write(text[-1])
    train_file.write(label)

for sample in test_samples:
    text, label = sample
    for i in range(len(text)-1):
        test_file.write(text[i])
        test_file.write('<TS>')
    test_file.write(text[-1])
    test_file.write(label)
    
train_file.close()
test_file.close()