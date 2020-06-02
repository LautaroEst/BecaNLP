import numpy as np
from sklearn.naive_bayes import MultinomialNB as sklearnMNB, BernoulliNB as sklearnBNB, CategoricalNB as sklearnCNB


class NaiveBayesClassifier(object):

	def __init__(self):
		if not hasattr(self,classifier):
			raise AttributeError('No se inicializó el modelo')

	def train(self,dataset):
		X, y = dataset
		self.classifier.fit(X,y)
		return self

	def predict(self,dataset):
		X, y = dataset
		y_predict = self.classifier.predict(X)
		return y, y_predict


class MultinomialNB(NaiveBayesClassifier):

	def __init__(self,**kwargs):
		self.classifier = sklearnMNB(**kwargs)


class BernoulliNB(NaiveBayesClassifier):
	# a esta clase le puedo pasar un umbral para recortar el valor en que considero
	# un valor binario.
	def __init__(self,**kwargs):
		self.classifier = sklearnBNB(**kwargs)

class CategoricalNB(NaiveBayesClassifier):

	def __init__(self,**kwargs):
		self.classifier = sklearnCNB(**kwargs)
