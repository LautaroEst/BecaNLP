import numpy as np
from sklearn.naive_bayes import MultinomialNB as sklearnMNB, BernoulliNB as sklearnBNB, CategoricalNB as sklearnCNB
from .BaseClassifiers import NaiveBayesClassifier


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
