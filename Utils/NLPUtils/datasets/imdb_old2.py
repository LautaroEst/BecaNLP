import sys, os
ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/IMDb')

from . import utils 

import re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset



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


def count_ngrams_and_vectorize(df,dev_size=.2,k_folds=None,
	random_state=0,labels_func=None,**kwargs):
	"""
	Función para vectorizar el dataframe y separarlo en train y dev. Si 
	k_folds es diferente a None, ignora el valor de dev_size y se devuelven
	los datasets correspondientes para hacer cross validation con k folds.
	"""
	N = len(df)
	indeces = utils.split_dev_kfolds(N,dev_size,k_folds,random_state)

	data = {}
	for i, (train_idx, dev_idx) in enumerate(indeces):
		df_train, df_dev = df.iloc[train_idx,:], df.iloc[dev_idx,:]
		vectorizer = utils.get_count_vectorizer(df_train['comment'],**kwargs)
		X_train = vectorizer.transform(df_train['comment'])
		X_dev = vectorizer.transform(df_dev['comment'])

		if labels_func is None:
			labels_func = lambda x: x

		y_train = labels_func(df_train['rate'].values)
		y_dev = labels_func(df_dev['rate'].values)

		data[i] = {'train': (X_train, y_train), 'dev': (X_dev, y_dev)}
	
	if k_folds is None:
		return data[0]

	return data








