import sys, os
ROOT_PATH = os.path.join(sys.path[0].split('Cursos')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/Stanford Sentiment Treebank/trees')

from collections import Counter
from itertools import chain
from .utils import ngrams
from nltk.tree import Tree
from sklearn.feature_extraction import DictVectorizer

import torch
from torch.utils.data import Dataset

def sentiment_treebank_reader(src_filename, class_func=None):
    """Iterator for the Penn-style distribution of the Stanford
    Sentiment Treebank. The iterator yields (tree, label) pairs.

    The labels are strings. They do not make sense as a linear order
    because negative ('0', '1'), neutral ('2'), and positive ('3','4')
    do not form a linear order conceptually, and because '0' is
    stronger than '1' but '4' is stronger than '3'.

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.
    class_func : None, or function mapping labels to labels or None
        If this is None, then the original 5-way labels are returned.
        Other options: `binary_class_func` and `ternary_class_func`
        (or you could write your own).

    Yields
    ------
    (tree, label)
        nltk.Tree, str in {'0','1','2','3','4'}

    """
    if class_func is None:
        class_func = lambda x: x
    with open(src_filename, encoding='utf8') as f:
        for line in f:
            tree = Tree.fromstring(line)
            label = class_func(tree.label())
            # As in the paper, if the root node doesn't fall into any
            # of the classes for this version of the problem, then
            # we drop the example:
            if label:
                for subtree in tree.subtrees():
                    subtree.set_label(class_func(subtree.label()))
                yield (tree, label)

def train_reader(**kwargs):
    """Convenience function for reading the train file, full-trees only."""
    src = os.path.join(DATASET_PATH, 'train.txt')
    return sentiment_treebank_reader(src, **kwargs)

def dev_reader(**kwargs):
    """Convenience function for reading the dev file, full-trees only."""
    src = os.path.join(DATASET_PATH, 'dev.txt')
    return sentiment_treebank_reader(src, **kwargs)

def test_reader(**kwargs):
    """Convenience function for reading the test file, full-trees only.
    This function should be used only for the final stages of a project,
    to obtain final results.
    """
    src = os.path.join(DATASET_PATH, 'test.txt')
    return sentiment_treebank_reader(src, **kwargs)


def binary_class_func(y):
	if y in ('0','1'):
		return '0'
	elif y in ('3','4'):
		return '1'

def ternary_class_func(y):
	if y in ('0','1'):
		return '0'
	elif y in ('3','4'):
		return '2'
	else:
		return '1'


def get_full_vocab():
	corpus = [tree.leaves() for tree, label in train_reader()]
	tk_to_freq = Counter(chain.from_iterable(corpus))
	return {tk: freq for tk, freq in sorted(tk_to_freq.items(), 
		key=lambda item: item[1], reverse=True)}






def get_bow_dataset(data=['train'],vocab=None,n_gram=1,n_classes=2):

	if n_classes not in (2,3,5):
		raise NameError('n_classes must be 2, 3 or 5')
	elif n_classes == 2:
		labels = ['0', '1']
	elif n_classes == 3:
		labels = ['0', '1', '2']
	else:
		labels = ['0', '1', '2', '3', '4']

	readers = []
	if 'train' in data:
		readers.append(train_reader)
	if 'dev' in data:
		readers.append(dev_reader)
	if 'test' in data:
		readers.append(test_reader)
	if len(readers) == 0:
		raise TypeError('Debe seleccionarse una o más opciones entre "train" "dev" o "test".')

	if isinstance(n_gram,int):
		if n_gram == 1:
			extract_features = lambda tree: Counter(tree.leaves())
		elif n_gram > 1:
			extract_features = lambda tree: Counter(ngram(tree.leaves(),n_gram))
		else:
			raise TypeError('n_gram tiene que ser mayor o igual a 1')
	elif isinstance(n_gram,tuple):
		low, high = n_gram
		if low >= high:
			raise TypeError('n_gram puede ser un entero o una  tupla (low, high) donde low < high')
		elif low == 1:
			extract_features = lambda tree: Counter(chain.from_iterable([tree.leaves()] + [ngram(tree.leaves(),i+1) for i in range(low,high)]))
		else:
			extract_features = lambda tree: Counter(chain.from_iterable([ngram(tree.leaves(),i) for i in range(low,high+1)]))


	vectorizer = DictVectorizer(sparse=True)
	vectorizer.fit([extract_features(tree) for tree, label in train_reader()])

	if vocab is not None:
		if isinstance(vocab,dict):
			vectorize.restrict(vocab.keys())
		elif isinstance(vocab,int):
			vectorize(list(get_full_vocab().keys())[:vocab])
		elif hasattr(vocab,'__iter__'):
			vectorize.restrict(vocab)
		else:
			raise TypeError('vocab must be an array-like object')


	dataset = {data_key: BOWSSTDataset(vectorizer,reader,extract_features) for data_key, reader in zip(data,readers)}
	dataset['vocab'] = vectorizer.vocabulary_
	dataset['labels'] = labels
	return dataset



class BOWSSTDataset(Dataset):

	def __init__(self,vectorizer,reader,extract_features):

		labels = []
		feat_dicts = []
		for tree, label in reader():
			labels.append(int(label))
			feat_dicts.append(extract_features(tree))
		
		self.feat_matrix = vectorizer.transform(feat_dicts)
		self.labels = torch.tensor(labels).view(-1,1)

	def __getitem__(self,idx):
		x = torch.from_numpy(self.feat_matrix[idx,:].toarray().reshape(-1))
		y = self.labels[idx]
		return x, y

	def __len__(self):
		return len(self.labels)







def count_ngrams_and_vectorize(labels_func=None,**kwargs):

	train_texts, train_labels = zip(*[(tree.leaves(), label) for tree, label in train_reader()])
	dev_texts, dev_labels = zip(*[(tree.leaves(), label) for tree, label in dev_reader()])
	vectorizer = utils.get_count_vectorizer(train_texts,**kwargs)

	X_train = vectorizer.transform(train_texts)
	X_dev = vectorizer.transform(dev_texts)

	if labels_func is None:
		labels_func = lambda x: x

	y_train = labels_func(train_labels)
	y_dev = labels_func(dev_labels)

	data = {'train': (X_train, y_train), 'dev': (X_dev, y_dev)}
	return data