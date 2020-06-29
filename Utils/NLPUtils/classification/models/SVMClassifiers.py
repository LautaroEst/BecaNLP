import numpy as np
from sklearn.svm import SVC, LinearSVC
from .BaseClassifiers import SVMClassifier


class SVCClassifier(SVMClassifier):

	def __init__(self,**kwargs):
		self.classifier = SVC(**kwargs)


class LinearSVCClassifier(SVMClassifier):

	def __init__(self,**kwargs):
		self.classifier = LinearSVC(**kwargs)


