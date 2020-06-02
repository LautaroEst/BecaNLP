import os
ROOT_PATH = os.path.join(__file__.split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/IMDb')


import re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


def vocab_file_reader():
	with open(os.path.join(DATASET_PATH,'imdb.vocab'),'r') as f:
		vocab = re.findall(r'[^\s]+',f.read())
	return vocab

def get_train_dataframe():
	df = pd.read_csv(os.path.join(DATASET_PATH,'train.csv'))
	return df

def get_test_dataframe():
	df = pd.read_csv(os.path.join(DATASET_PATH,'test.csv'))
	df = df.rename(index)
	return df

def get_unsup_dataseries():
	df = pd.read_csv(os.path.join(DATASET_PATH,'train_unsup_all.csv'))
	return df.sort_index()['comment']

