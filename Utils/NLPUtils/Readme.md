# NLPUtils

Este es un módulo para hacer programas y experimentos sobre NLP. 

## Estructura

```
NLPUtils
   |
   |--- Classification: Modelos, criterios de evaluación y vectorización de datasets para clasificación
   |
   |--- Datasets: Funciones para obtener los datasets disponibles
   |
   |--- feature_extraction: Métodos y algoritmos para extraer features de textos. Incluye: Bag-of-ngrams, algoritmos distribucionales de word vectors.
   |
   |--- preprocessing: funciones handlers para preprocesar el texto

```

1. Classification: La estructura general de todos los modelos consiste en

```Python
class Classifier:

	# ...

	def train(...):
		# Entrenamiento del modelo

	def predict(...):
		# Nuevas predicciones
```

Notamos que esta estructura puede corresponder a un modelo paramétrico como no paramétrico, con la diferencia de que en el no paramétrico la función `train()` no hace más guardar las muestras de entrenamiento. 

El submódulo `models` contiene varios clasificadores estandarizados de los cuales pueden derivarse los demás. Para ello, se definió la clase `NeuralNetClassifier`, base de todos los clasificadores paramétricos neuronales (lineales, convolucionales, recurrentes, etc.). Todas las redes neuronales diseñadas serán subclases de esta clase. Algunos de estos modelos definidos hasta el momento son:

* Modelos lineales: `LogisticRegressionClassifier`, `LinearSoftmaxClassifier`. Todos estos modelos son subclases directas de `NeuralNetClassifier`.

* Modelos secuenciales: `ManyToOneRecurrentClassifier`, ... Todos estos modelos son subclases de `SequenceClassifier`, que es la clase base de todos los clasificadores de secuencias. Esta clase es simplemente una subclase de `NeuralNetClassifier` con las funciones `train()` y `predict()` adaptadas al entrenamiento con muestras de secuencias. Se puede usar de la misma manera que se usa `NeuralNetClassifier`.

Los métodos de evaluación de los classificadores anteriores incluyen:

* Accuracy
* Precision, Recall
* Confusion Matrix
* Micro, Macro y weighted f_beta score.

También se incluyen los objetos tipo `Vectorizers` que convierten un dataset supervisado de texto en un conjunto de vectores con sus respectivos labels, listos para entrenar el modelo. Estos incluyen:

* `BagOfNgramVecotizer`: Utiliza un CountVecotirizer (ya sea el de sklearn o uno propio) para convertir texto vectores de cuentas.
* `WordVectorsVectorizer`: Convierte el texto en una secuencia de word embeddings preentrenados.
* Otros...

2. Datasets: Funciones para preprocesar y cargar los datasets ubicados en la carpeta `BecaNLP/Utils/Datasets/`. Cada dataset es un caso aparte y tiene sus propias funciones para ser procesado. Además, hay varias formas de definir el objeto `Dataset` que se va a usar en el entrenamiento para el mismo dataset, y todas estas están en un mismo archivo. 

3. feature_extraction: Algoritmos para extraer features de un texto: word2vec, glove, coocurrence, bag-of-ngrams, etc.
