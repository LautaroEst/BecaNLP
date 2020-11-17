import os
ROOT_PATH = os.path.join(__file__.split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/Stanford Sentiment Treebank/trees')

from nltk.tree import Tree
import pandas as pd


def treebank_reader(src_file,input_fn=None,label_fn=None):
    if input_fn is None:
        input_fn = lambda x: x
    if label_fn is None:
        label_fn = lambda x: x
    with open(src_file, encoding='utf8') as f:
        for line in f:
            tree = Tree.fromstring(line)
            label = tree.label()
            if label is not None:
                yield input_fn(tree), label_fn(label)


def train_reader(input_fn=None,label_fn=None):
    src = os.path.join(DATASET_PATH, 'train.txt')
    return treebank_reader(src, input_fn, label_fn)


def dev_reader(input_fn=None,label_fn=None):
    src = os.path.join(DATASET_PATH, 'dev.txt')
    return treebank_reader(src, input_fn, label_fn)


def test_reader(input_fn=None,label_fn=None):
    src = os.path.join(DATASET_PATH, 'test.txt')
    return treebank_reader(src, input_fn, label_fn)


def as_frame(data='train',n_classes=2):
    if data == 'train':
        reader = train_reader
    elif data == 'dev':
        reader = dev_reader
    elif data == 'test':
        reader = test_reader
    else:
        raise TypeError('Dataset not correct.')

    if n_classes not in (2,3,5):
        raise TypeError('Number of classes must be 2, 3 or 5')

    input_fn = lambda x: x.leaves()
    label_fn = lambda x: int(x)

    return pd.DataFrame(reader(input_fn,label_fn),columns=['comment','rate'])