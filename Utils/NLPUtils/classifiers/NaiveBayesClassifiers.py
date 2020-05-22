import numpy as np
from sklearn.naive_bayes import MultinomialNB as sklearnMNB, BernoulliNB as sklearnBNB

class MultinomialNB(sklearnMNB):
	pass

class BernoulliNB(sklearnBNB):
	# a esta clase le puedo pasar un umbral para recortar el valor en que considero
	# un valor binario.
	pass
