import numpy as np
from sklearn.svm import SVC, LinearSVC

class SVMClassifier(object):

	def __init__(self):
		if not hasattr(self,classifier):
			raise AttributeError('No se inicializó el modelo')


	def train(self,dataset,sample_weight=None):
		X, y = dataset
		self.classifier.fit(X,y,sample_weight)
		return self

	def predict(self,dataset):
		X, y = dataset
		y_predict = self.classifier.predict(X)
		return y, y_predict


class SVCClassifier(SVMClassifier):

	def __init__(self,**kwargs):
		self.classifier = SVC(**kwargs)


class LinearSVCClassifier(SVMClassifier):

	def __init__(self,**kwargs):
		self.classifier = LinearSVC(**kwargs)


