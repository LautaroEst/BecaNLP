# NLPUtils

Este es un módulo para cargar todos los programas hechos hasta el momento sobre NLP. 

## Estructura

```
NLPUtils
   |
   |--- Classifiers: Modelos para clasificación
   |
   |--- Datasets: Funciones para obtener los datasets disponibles
   |
   |--- Módulos de aplicaciones específicas
   			|
   			|--- VSM: Implementación de aplicaciones los algoritmos para hacer espacios de modelos semánticos.
   			|
   			|--- LM: Implementación de algoritmos de modelo de lenguaje. 
   			|
   			|--- Otros: ...

```

1. Classifiers: Modelos utilizados para clasificación. La estructura general de todos los modelos consiste en

```Python
class Classifier:

	# ...

	def train(...):
		# Entrenamiento del modelo

	def predict(...):
		# Nuevas predicciones
```

Notamos que esta estructura puede corresponder a un modelo paramétrico como no paramétrico, con la diferencia de que en el no paramétrico la función `train()` no hace más guardar las muestras de entrenamiento. 

El submódulo `classifiers` contiene varios clasificadores estandarizados de los cuales pueden derivarse los demás. Para ello, se definió la clase `NeuralNetClassifier`, base de todos los clasificadores paramétricos neuronales (lineales, convolucionales, recurrentes, etc.). Todas las redes neuronales diseñadas serán subclases de esta clase. Algunos de estos modelos definidos hasta el momento son:

* Modelos lineales: `LogisticRegressionClassifier`, `LinearSoftmaxClassifier`. Todos estos modelos son subclases directas de `NeuralNetClassifier`.

* Modelos secuenciales: `ManyToOneRecurrentClassifier`, ... Todos estos modelos son subclases de `SequenceClassifier`, que es la clase base de todos los clasificadores de secuencias. Esta clase es simplemente una subclase de `NeuralNetClassifier` con las funciones `train()` y `predict()` adaptadas al entrenamiento con muestras de secuencias. Se puede usar de la misma manera que se usa `NeuralNetClassifier`.

2. Datasets: Funciones para preprocesar y cargar los datasets ubicados en la carpeta `BecaNLP/Utils/Datasets/`. Cada dataset es un caso aparte y tiene sus propias funciones para ser procesado. Además, hay varias formas de definir el objeto `Dataset` que se va a usar en el entrenamiento para el mismo dataset, y todas estas están en un mismo archivo. 

A su vez, existe un archivo `utils.py` que contiene funciones comunes utilizadas por todos los datasets, como la tokenización o la transformación a BOW.

3. Módulos de aplicaciones específicas: VSM. En este módulo se implementan algoritmos de espacios de modelos semánticos como word2vec, glove o métodos de co-ocurrencia. También hay una parte del módulo dedicada a cargar vectores ya preentrenados.

4. Módulos de aplicaciones específicas: Language Modeling. Se implementan los modelos que sirven para modelos de lenguaje. 