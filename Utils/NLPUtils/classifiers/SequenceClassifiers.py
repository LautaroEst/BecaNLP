import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NeuralNetClassifier


class SequenceClassifier(NeuralNetClassifier):

	def _pad_collate_fn(data_batch):
		x_batch, y_batch = zip(*data_batch)
		x_lenghts = torch.tensor([sample.size(0) for sample in x_batch])
		sorted_lenghts_idx = torch.argsort(x_lenghts,descending=True)
		x_lenghts = x_lenghts[sorted_lenghts_idx]
		x_batch = nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
		x_batch = x_batch[sorted_lenghts_idx]
		y_batch = y_batch[sorted_lenghts_idx]
		# y_batch tiene que tener por lo menos 2 dimensiones (el batch primero)
		y_lenghts = torch.tensor([sample.size(0) for sample in y_batch])
		return x_batch, x_lenghts, y_batch, y_lenghts

	def train(self, train_dataset, optim_algorithm='SGD', 
        	  epochs=1, batch_size=512, **kwargs):
		"""
		Función para entrenar el modelo.
		"""

		model = self.model
		device = self.device

		# Definimos el dataloader: 
		loader = DataLoader(train_dataset, batch_size, 
			shuffle=True, collate_fn=_pad_collate_fn)

		# Seleccionamos el método de optimización:
		try:
		    optimizer = self.optimizer
		except AttributeError:
		    if optim_algorithm == 'SGD':
		        optimizer = optim.SGD(model.parameters(), **kwargs)
		    elif optim_algorithm == 'Adam':
		        optimizer = optim.Adam(model.parameters(), **kwargs)
		    else:
		        raise TypeError('Algoritmo de optimización no soportado')
		model.train()

		try:
		    current_epoch = self.current_epoch
		except AttributeError:
		    current_epoch = 1

		# Inicializamos el historial de la loss:
		try:
		    loss_history = self.loss_history
		    print('Resuming training from epoch {}...'.format(current_epoch))
		except AttributeError: 
		    print('Starting training...')
		    loss_history = []
		    
		# Comenzamos el entrenamiento:
		try:
		    
		    for e in range(current_epoch, current_epoch+epochs):
		        for t, (x, x_lenghts, y, y_lenghts) in enumerate(loader):
		            x = x.to(device)
		            y = y.to(device)
		            
		            optimizer.zero_grad() # Llevo a cero los gradientes de la red
		            scores = model(x,x_lenghts) # Calculo la salida de la red
		            loss = self.loss(scores,x_lenghts,y,y_lenghts) # Calculo el valor de la loss
		            loss.backward() # Calculo los gradientes
		            optimizer.step() # Actualizo los parámetros
		        
		            loss_history.append(loss.item())
		            
		        print('Epoch {} finished. Approximate loss: {:.4f}'.format(e, sum(loss_history[-5:])/5))
		            
		    print('Training finished')
		    print()            

		except KeyboardInterrupt:
		    print('Exiting training...')
		    print()
		    
		self.model = model
		self.loss_history = loss_history
		self.optimizer = optimizer
		self.current_epoch = e + 1

	def loss(scores,scores_lenghts,target,target_lenghts):
		pass





class ManyToOneRecurrentClassifier(SequenceClassifier):
	"""
	Implementación de un modelo end-to-end recurrente (Vanilla, LSTM o GRU).
	"""
	class Model(nn.Module):

		def __init__(self,rnn,n_classes,*args,**kargs):
			if rnn == 'vanilla':
				self.rnn = nn.RNN(*args,**kwargs)
			elif rnn == 'lstm':
				self.rnn = nn.LSTM(*args,**kwargs)
			elif rnn == 'gru':
				self.rnn = nn.GRU(*args,**kwargs)
			else:
				raise NameError('Not supported {} recurrent model'.format(rnn))
			self.linear = nn.Linear(args[1],n_classes)

		def forward(x,x_lenghts):
			x = nn.utils.rnn.pack_padded_sequence(x,batch_first=True,lenghts=x_lenghts)
			x = self.rnn(x)
			x = self.linear(x.data)
			return x


	def __init__(self,device,rnn='vanilla',*args,**kwargs):
		model = self.Model(rnn,*args,**kwargs)
		super().__init__(model,device)

	def loss(self,scores,target):
		return F.cross_entropy(scores,target,reduce='mean')




