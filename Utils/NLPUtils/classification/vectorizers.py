from ..feature_extraction import CorpusCountVectorizer
import torch
from torch.utils.data import Dataset


class _torch_sparse_matrix(Dataset):

		def __init__(self,X,y):
			self.X = X
			self.y = y

		def __getitem__(self,idx):
			return torch.from_numpy(self.X[idx,:].toarray()), torch.tensor([self.y[idx]])

		def __len__(self):
			return len(self,y)



class BagOfNgramsVectorizer(CorpusCountVectorizer):

	"""
	Vectorizer para DataFrames que contienen en la primera
	columna las muestras de entrada sin vectorizar y en la 
	segunda, los labels.
	"""

	def __init__(self,**kwargs):
		self.istorch = kwargs.pop('istorch',False)
		self._reweight = kwargs.pop('reweight',None)
		self._label_fn = kwargs.pop('label_fn',None)
		super().__init__(**kwargs)


	def fit_transform(self,data_df):
		X = super().fit_transform(data_df.iloc[:,0])
		y = data_df.iloc[:,1].values.copy()

		X = self.do_reweight(X,self._reweight)
		y = self.apply_label_fn(y,self._label_fn)

		dataset = _torch_sparse_matrix(X,y) if self.istorch else (X, y)
		return dataset


	def transform(self,data_df):
		X = super().transform(data_df.iloc[:,0])
		y = data_df.iloc[:,1].values.copy()

		X = self.do_reweight(X,self._reweight)
		y = self.apply_label_fn(y,self._label_fn)
		
		dataset = _torch_sparse_matrix(X,y) if self.istorch else (X, y)
		return dataset


	@staticmethod
	def do_reweight(X,reweight):

		if reweight is not None:
			if reweight == 'tfidf':
				pass
			elif reweight == 'ppmi':
				pass
			elif reweight == 'oe':
				pass
			else:
				raise TypeError('Reweight not supported')

		return X


	@staticmethod
	def apply_label_fn(y,label_fn):

		if label_fn is not None:
			y = label_fn(y)

		return y


# class SequenceVectorizer(object):

# 	def __init__(self,vocabulary=None,tokenizer=r'\b\w+\b',max_vocab=None):
# 		if vocabulary is None:
# 			self.vocabulary = {}
# 			self.max_vocab = max_vocab 

# 		if isinstance(tokenizer,str):
# 			self.tokenizer = _build_tokenizer_from_regex(tokenizer)

# 	def fit_transform(self,corpus):

# 		indices = []
# 		lenghts = []

# 		for doc in corpus:
# 			tokens = self.tokenizer(doc)
# 			idx = [self.vocabulary.get(token,-1) for token in tokens]
# 			indices.append(idx)
# 			lenghts.append(len(idx))
		

# 	@staticmethod
# 	def _build_tokenizer_from_regex(pattern):
# 		patt = re.compile(pattern)
# 		return patt.findall

