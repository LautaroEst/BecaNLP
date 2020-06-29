from sklearn.dummy import DummyClassifier


class MostFrequentClassifier(DummyClassifier):
	"""
	Busca qué clase es la más frecuente y clasifica todas las muestras a esa clase.
	"""
	def __init__(self,**kwargs):
		super().__init__(strategy='most_frequent',**kwargs)


class UniformClassifier(DummyClassifier):
	"""
	Genera predicciones aleatorias según la distribución de las clases.
	"""
	def __init__(self,**kwargs):
		super().__init__(strategy='most_frequent',**kwargs)



