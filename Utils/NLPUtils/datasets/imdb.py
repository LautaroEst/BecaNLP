import sys, os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/IMDb')

from . import utils 

import re
import pandas as pd
import numpy as np


def vocab_file_reader():
	with open(os.path.join(DATASET_PATH,'imdb.vocab'),'r') as f:
		vocab = re.findall(r'[^\s]+',f.read())
	return vocab

def train_reader():
	df = pd.read_csv(os.path.join(DATASET_PATH,'train.csv'),
		names=['comment','rate'],dtype={'comment':str,'rate':np.int},skiprows=1)
	return df

def test_reader():
	df = pd.read_csv(os.path.join(DATASET_PATH,'test.csv'),
		names=['comment','rate'],dtype={'comment':str,'rate':np.int},skiprows=1)
	return df

def unsup_reader():
	df = pd.read_csv(os.path.join(DATASET_PATH,'train_unsup_all.csv'),
		names=['idx','comment'],dtype={'idx':np.int,'comment':str,},skiprows=1)
	return df['comment']