import torch
import torch.nn as nn



class WordVectorsTrainer(object):


	def __init__(self,samples,vocab,model,device,loss):
		device, model = self._select_device(device, model)
		self.device = device
		self.model = model.to(device)
		self.samples = samples
		self.vocab = vocab
		self.loss = loss


	def train(self, epochs=1, verbose=True, optim_algorithm='minibatch', **kwargs):
		"""
		Función para entrenar el modelo.
		"""

		model = self.model
		device = self.device

		# Definimos el dataloader:
		if optim_algorithm == 'SGD':
			batch_size = 1
		else:
			batch_size = kwargs.pop('batch_size',512)
		loader = DataLoader(train_dataset, batch_size, shuffle=True)

		# Seleccionamos el método de optimización:
		try:
			optimizer = self.optimizer
		except AttributeError:
			if optim_algorithm == 'minibatch' or optim_algorithm == 'SGD':
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
		zero_grad = optimizer.zero_grad
		step = optimizer.step
		loss_fn = self.loss

		try:

			for e in range(current_epoch, current_epoch+epochs):
				for t, (x,y) in enumerate(loader):
					x = x.to(device)
					y = y.to(device)

					zero_grad() # Llevo a cero los gradientes de la red
					scores = model(x) # Calculo la salida de la red
					loss = loss_fn(scores,y) # Calculo el valor de la loss
					loss.backward() # Calculo los gradientes
					step() # Actualizo los parámetros

					loss_history.append(loss.item())

				if verbose:
					print('Epoch {} finished. Approximate loss: {:.4f}'.format(e, sum(loss_history[-5:])/len(loss_history[-5:])))


			print('Training finished')
			print()

		except KeyboardInterrupt:
			print('Exiting training...')
			print()

		self.model = model
		self.loss_history = loss_history
		self.optimizer = optimizer
		self.current_epoch = e + 1


	@staticmethod
	def _select_device(device, model):
		if device is None:
			device = torch.device('cpu')
			print('Warning: Dispositivo no seleccionado. Se utilizará la cpu.')
		elif device == 'parallelize':
			if torch.cuda.device_count() > 1:
				device = torch.device('cuda:0')
				model = nn.DataParallel(model)
			else:
				device = torch.device('cpu')
				print('Warning: No es posible paralelizar. Se utilizará la cpu.')
		elif device == 'cuda:0' or device == 'cuda:1':
			if torch.cuda.is_available():
				device = torch.device(device)
			else:
				device = torch.device('cpu')
				print('Warning: No se dispone de dispositivos tipo cuda. Se utilizará la cpu.')
		elif device == 'cpu':
			device = torch.device(device)
		else:
			raise RuntimeError('No se seleccionó un dispositivo válido')

		return device, model


	def save_checkpoint(self, filename):
		print('Saving checkpoint to file...',end=' ')
		model = self.model.to(torch.device('cpu'))
		torch.save({
			'epoch': self.current_epoch,
			'model_state_dict': model.state_dict(),
			'optimizer': self.optimizer,
			'loss': self.loss_history
			}, filename)
		print('OK')


	def load_checkpoint(self, filename):
		print('Loading checkpoint from file...',end=' ')
		checkpoint = torch.load(filename)
		self.current_epoch = checkpoint['epoch']
		model = self.model
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model = model.to(self.device)
		self.optimizer = checkpoint['optimizer']
		self.loss_history = checkpoint['loss']
		print('OK')


	def save_parameters(self, filename):
		print('Saving parameters to file...',end=' ')
		model = self.model.to(torch.device('cpu'))
		torch.save(model.state_dict(), filename)
		print('OK')


	def load_parameters(self, filename):
		print('Loading parameters from file...',end=' ')
		model = self.model
		model.load_state_dict(torch.load(filename))
		self.model = model.to(self.device)
		print('OK')