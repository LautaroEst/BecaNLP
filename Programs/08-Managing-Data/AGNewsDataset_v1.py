import os
import pandas as pd
import re

import torch


class AGNewsDataset(object):
    
    token_pad = '<PAD>'
    token_unk = '<UNK>'
    token_sep = '<TS>'
    special_tokens = [token_pad, token_unk]
    
    def __init__(self, root, train=True, use_test_tokens=False):
        
        if not os.path.exists(root):
            raise IOError('Carpeta {} no encontrada'.format(root))
        self.root = root + '/' if root[-1] != '/' else root
        
        print('Buscando archivos train.csv test.csv...')
        self.train_path = '{}train.csv'.format(self.root)
        self.test_path = '{}test.csv'.format(self.root)
        if not os.path.exists(self.train_path):
            raise IOError('Archivo train.csv no encontrado en el directorio {}'.format(self.train_path))
        elif not os.path.exists(self.train_path):
            raise IOError('Archivo test.csv no encontrado en el directorio {}'.format(self.test_path))
        
        self.preprocessed_train_path = '{}preprocessed_train.csv'.format(self.root)
        self.preprocessed_test_path = '{}preprocessed_test.csv'.format(self.root)
        if not (os.path.exists(self.preprocessed_train_path) and os.path.exists(self.preprocessed_test_path)):
            print('Preprocesando archivos train.csv y test.csv...')
            self._preprocess_data()
        else:
            print('Archivos train.csv y test.csv encontrados y preprocesados.')
        
        self._data = pd.read_csv(self.preprocessed_train_path) if train else pd.read_csv(self.preprocessed_test_path)
        
        max_len = 0
        for index, row in self._data.iterrows():
            lenght = len(self._string_to_tokens(row['Title']))
            if max_len < lenght:
                max_len = lenght
        self.max_len = max_len
        
        self.vocabulary = self._get_vocabulary(use_test_tokens=use_test_tokens)
        
        
        #### TO DO: #####
        for index, row in self._data.iterrows():
            title = self._string_to_tokens(row['Title'])
            new_title = ''
            for word in title:
                if self.vocabulary.get_freq(word) < 50:
                    new_title = token_sep.join([new_title, self.token_unk])
                else:
                    new_title = token_sep.join([new_title, word])
            row['Title'] = new_title
            self._data.iloc[index] = row
        ##################
        
    def __getitem__(self, idx):
                
        if isinstance(idx,torch.Tensor):
            index = idx.tolist()
        else:
            index = idx
        try:
            text, cls_idx = self._data.iloc[index,:]
        except IndexError:
            raise IndexError('{} exceeds index of dataset'.format(index))
            return
        
        cls_idx = torch.tensor(cls_idx - 1, dtype=torch.long)
        
        text = self._string_to_tokens(text)
        text_idx = torch.tensor([self.vocabulary.token_to_index(word) for word in text], dtype=torch.long)
        text_idx = torch.nn.functional.pad(text_idx, 
                                       pad=(0,self.max_len - len(text_idx)),
                                       mode='constant', 
                                       value=self.vocabulary.token_to_index(self.token_pad))
        
        return text_idx, cls_idx
    

    def __len__(self):
        return len(self._data)
    

    def _preprocess_data(self):
        """
        Toma los archivos que componen el corpus, los preprocesa
        y devuelve en dos archivos .csv (train y test) las muestras
        preprocesadas.
        """
        print('Creando preprocessed_train.csv y preprocessed_test.csv...')
        for filename, pp_filename in [(self.train_path, self.preprocessed_train_path), (self.test_path, self.preprocessed_test_path)]:
            df = pd.read_csv(filename,header=None)
            with open(pp_filename, 'w+') as pp_f:
                pp_f.write('Title,Class label\n')
                for index, row in df.iterrows():
                    title = re.sub( r'"', r"'", row[1])
                    new_row = re.sub( r' ', self.token_sep, '\"{0}\",{1:}\n'.format(title,int(row[0])) )
                    pp_f.write(new_row)
        print('OK!')
        
        
    def _get_vocabulary(self, use_test_tokens=False):
        
        vocabulary = AGNewsVocabulary()
        
        
        for token in self.special_tokens:
            idx = vocabulary.add_token(token)
            vocabulary._idx_to_freq[idx] -= 1
        
        if use_test_tokens:
            filenames = [self.preprocessed_train_path, self.preprocessed_test_path]
        else:
            filenames = [self.preprocessed_train_path]
            
        for filename in filenames:
            file_df = pd.read_csv(filename)
            for title in file_df.iloc[:,0]:
                title = self._string_to_tokens(title)
                for word in title:
                    vocabulary.add_token(word)
            
        return vocabulary
    
    def _string_to_tokens(self,string):
        return string.split(self.token_sep)




class AGNewsVocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self):

        self._token_to_idx = {}
        self._idx_to_token = {}
        self._idx_to_freq = {}

    def add_token(self, token):
        
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
            self._idx_to_freq[index] += 1
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            self._idx_to_freq[index] = 1
        return index
    
    def index_to_token(self, index):
        
        if not isinstance(index, list):
            if not isinstance(index, int):
                raise NameError("'index' must be an integer or list of integers")
            if index not in self._idx_to_token:
                raise KeyError('the index {} exeeds the Vocabulary lenght'.format(index))
            return self._idx_to_token[index]
        
        tokens = []
        for idx in index:
            if not isinstance(idx, int):
                raise NameError("{} is not an integer".format(idx))
            if idx not in self._idx_to_token:
                raise KeyError('the index {} exeeds the Vocabulary lenght'.format(idx))
            tokens.append(self._idx_to_token[idx])
        return tokens

    def token_to_index(self, token):
        
        if not isinstance(token, list):
            if not isinstance(token, str):
                raise NameError("'token' must be a string or list of strings")
            if token not in self._token_to_idx:
                raise KeyError('the token {} is not in the Vocabulary'.format(token))
            return self._token_to_idx[token]
        
        indeces = []
        for tk in token:
            if not isinstance(tk, str):
                raise NameError("'token' must be a string or list of strings")
            if tk not in self._token_to_idx:
                raise KeyError('the token {} is not in the Vocabulary'.format(tk))
            indeces.append(self._token_to_idx[tk])
        return indeces
    
    def get_freq(self, token_or_index):
        freqs = []
        try:
            length = len(token_or_index)
        except TypeError:
            tk_or_idx_list = [token_or_index]
        
        for tk_or_idx in tk_or_idx_list:
            if isinstance(tk_or_idx, int):
                if tk_or_idx not in self._idx_to_token:
                    raise KeyError('the index {} exeeds the Vocabulary lenght'.format(tk_or_idx))
                freqs.append(self._idx_to_freq[tk_or_idx])
            if isinstance(tk_or_idx, str):
                if tk_or_idx not in self._token_to_idx:
                    raise KeyError('the token {} is not in the Vocabulary'.format(tk_or_idx))
                freqs.append(self._idx_to_freq[self._token_to_idx[tk_or_idx]])
            raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        if len(freqs) == 1 and not isinstance(token_or_index, list):
            return freqs[0]
        return freqs

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)