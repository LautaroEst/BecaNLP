import numpy as np
from scipy.sparse import csr_matrix

from .BaseClassifiers import NaiveBayesClassifier


class MultinomialNB(object):

	def __init__(self,n_features,n_classes,n_bins=None):
		self.n_features = n_features
		self.n_bins = n_bins
		self.n_classes = n_classes
		

	def train(self,X_train,y_train,alpha=1.):

		if isinstance(X_train,csr_matrix):
			X_train = X_train.toarray().copy()
		else:
			X_train = X_train.copy()

		if self.n_bins is None:
			uniques_values = np.uniques(X_train)
			for i, val in enumerate(uniques_values):
				X_train[X_train == val] = i	
			self.n_bins = len(uniques_values)
		
		hist, bins_edges = np.histogram(X_train.reshape(-1),bins=self.n_bins)
		X_train = np.digitize(X_train,bins_edges) - 1

		log_probs = np.ones((n_classes,n_features,n_bins))
		for i in range(n_features):
			for j in range(n_classes):
				mask = y_train == j
				hist, bins_edges = np.histogram(X_train[mask,i],bins=self.n_bins)
				log_probs[j,i,:] = np.log(hist + alpha) - np.log(len(mask) + n_features * alpha)

		self.log_probs = log_probs

	def predict(self,X_test):
		pass







