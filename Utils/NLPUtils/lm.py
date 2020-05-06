import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifiers import NeuralNetClassifier, SequenceClassifier


class LMSeqClassifier(SequenceClassifier):

	class Model(nn.Module):

		def __init__(self,vocab_size,emb_dim,hidden_dim,**kwargs):
			super().__init__()
			self.embedding = nn.Embedding(vocab_size,emb_dim)
			self.rnn = nn.RNN(emb_dim,hidden_dim,**kwargs)
			self.linear = nn.Linear(hidden_dim,vocab_size)

		def forward(x,x_lenghts):
			x = self.embedding(x)
			x = nn.utils.rnn.pack_padded_sequence(x,batch_first=True,lenghts=x_lenghts)
			x = self.rnn(x)
			x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
			x = [self.linear(sample[:x_lenghts[i]]) for i,sample in enumerate(x)]

	def __init__(self,device,vocab_size,emb_dim,hidden_dim,**kwargs):
		model = self.Model(vocab_size,emb_dim,hidden_dim,**kwargs)
		super().__init__(model,device)




