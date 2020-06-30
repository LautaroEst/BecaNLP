import torch
import torch.nn as nn

from .neural_base import WordVectorsTrainer


class CBOWModel(nn.Module):

	def __init__(self,vocab_size,embeddings_dim):
		pass

	def forward(self,x):
		pass

class SkipGramModel(nn.Module):

	def __init__(self,vocab_size,embeddings_dim):
		super().__init__()
		self.linear = nn.Linear(1,1)

	def forward(self,x):
		return x

class SkipGramNSModel(nn.Module):

	def __init__(self,vocab_size,embeddings_dim):
		pass

	def forward(self,x):
		pass


def get_skip_gram_samples(corpus, left_n=2, right_n=2, tokenizer=None, 
						min_count=0., max_count=None, max_words=None):
	return None, {}

def get_skip_gram_ns_samples(corpus, left_n=2, right_n=2, tokenizer=None, 
							min_count=0., max_count=None, max_words=None):
	return None, {}

def get_cbow_samples(corpus, left_n=2, right_n=2, tokenizer=None, 
					min_count=0., max_count=None, max_words=None):
	return None, {}


def skip_gram_loss(scores,target):
	pass

def skip_gram_ns_loss(scores,target):
	pass

def cbow_loss(scores,target):
	pass



class Word2VecTrainer(WordVectorsTrainer):

	def __init__(self, corpus, left_n=2, right_n=2, tokenizer=None, min_count=0., 
		max_count=None, max_words=None, algorithm='SkipGram', embeddings_dim=100, device='cpu'):
		
		model_class, get_samples = self._select_model(algorithm)
		samples, vocab = get_samples(corpus,left_n,right_n,tokenizer,min_count,max_count,max_words)		
		super().__init__(samples,vocab,model_class(len(vocab),embeddings_dim),device,loss)


	@staticmethod
	def _select_model(algorithm):

		if algorithm == 'SkipGram':
			model_class = SkipGramModel
			get_samples = get_skip_gram_samples
			loss = skip_gram_loss
		elif algorithm == 'SkipGram-ns':
			model_class = SkipGramNSModel
			get_samples = get_skip_gram_ns_samples
			loss = skip_gram_ns_loss
		elif algorithm == 'CBOW':
			model_class = CBOWModel
			get_samples = get_cbow_samples
			loss = cbow_loss
		else:
			raise TypeError('Algorithm not supported.')

		return model_class, get_samples, loss