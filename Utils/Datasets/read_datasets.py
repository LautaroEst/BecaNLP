import pandas as pd
import numpy as np
import sys
import os
import re

ROOT_PATH = os.path.join(sys.path[0].split('BecaNLP')[0],'BecaNLP')
DATASETS_ROOT_PATH = os.path.join(ROOT_PATH,'Utils/Datasets')

def read_AG_dataset():
    DATASET_PATH = os.path.join(DATASETS_ROOT_PATH,'AG')
    with open(os.path.join(DATASET_PATH,'classes.txt'),'r') as f:
        categories = re.split(r'\s+', f.read())
        categories = list(filter(lambda item: item != '',categories))
    df_train = pd.read_csv(os.path.join(DATASET_PATH,'train.csv'),
                           names=['label','title','description'], 
                           dtype={'label':np.int,'title':str,'description':str})
    df_test = pd.read_csv(os.path.join(DATASET_PATH,'test.csv'),
                          names=['label','title','description'], 
                          dtype={'label':np.int,'title':str,'description':str})
    for df in [df_train, df_test]:
        df['label'] = df['label'] - 1
        df['title'] = df['title'].str.replace(r' ?\(.+\) ?$', '', regex=True)
        df['description'] = df['description'].str.replace(r'^.+ -\s?', '', regex=True)
    return df_train, df_test, categories


def read_imdb_dataset():
    DATASET_PATH = os.path.join(DATASETS_ROOT_PATH,'IMDb')
    with open(os.path.join(DATASET_PATH,'imdb.vocab'),'r') as f:
        vocab = re.split(r'\s+',f.read())
        vocab = list(filter(lambda item: item != '',vocab))
    
#     df_train_pos = pd.read_csv(os.path.join(DATASET_PATH,'train_pos_all.csv'))
#     df_train_neg = pd.read_csv(os.path.join(DATASET_PATH,'train_neg_all.csv'))
#     df_test_pos = pd.read_csv(os.path.join(DATASET_PATH,'test_pos_all.csv'))
#     df_test_neg = pd.read_csv(os.path.join(DATASET_PATH,'test_neg_all.csv'))
    
#     df_train = pd.concat([df_train_pos.iloc[:,1:],df_train_neg.iloc[:,1:]]).sample(frac=1)
#     df_test = pd.concat([df_test_pos.iloc[:,1:],df_test_neg.iloc[:,1:]]).sample(frac=1)

    df_train = pd.read_csv(os.path.join(DATASET_PATH,'train.csv'),names=['comment','rate'],
                           dtype={'comment':str,'rate':np.int},skiprows=1)
    df_test = pd.read_csv(os.path.join(DATASET_PATH,'test.csv'),names=['comment','rate'],
                          dtype={'comment':str,'rate':np.int},skiprows=1)
    
    return df_train, df_test, vocab
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
