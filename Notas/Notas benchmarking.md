# Evaluación de tareas en NLP

¿Cómo evaluamos tareas en NLP? A continuación se muestran notas de algunos papers relevantes en este tema. Entre ellos:

* [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://openreview.net/pdf?id=rJ4km2R5t7)

* [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://w4ngatang.github.io/static/papers/superglue.pdf) 

* [jiant: A Software Toolkit for Research on General-Purpose Text Understanding Models](https://arxiv.org/abs/2003.02249)

* [Spanish Pre-trained BERT model and Evaluation Data](https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf)

En cada uno de estos papers se refieren a *tasks* o tareas, pero técnicamente cada *task* incluye, además de la tarea en sí, un dataset específico y una medida de desempeño.


## Tareas utilizadas en GLUE

Single-sentence tasks:

* CoLA: Clasificación binaria sobre el dataset [*The Corpus of Linguistic Acceptability*](http://nyu-mll.github.io/CoLA), que consiste en una serie de ejemplos de frases en inglés y si son gramaticalmente posibles o no. La medida de desempeño de este algoritmo es el coeficiente de Mathiews.

* SST-2: Análisis de sentimientos binario (positivo/negativo) sobre el dataset [*Stanford Sentiment Analysis*](https://nlp.stanford.edu/sentiment/). La medida de desempeño es accuracy.

Similarity and Paraphrase tasks:

* MRPC: Clasificación de pares de oraciones sobre el dataset [The Microsoft Research Paraphrase Corpus](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py#L8) que contiene pares de oraciones con su correspondiente etiqueta de si son semánticamente equivalentes o no. Se usa F1 y accuracy para la medida del desempeño. Esta es la tarea de paraphrase.

* QQP: Clasificación de pares de preguntas sobre el dataset [Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs), que es igual que el MRPC pero con preguntas. También se usa F1 y accuracy y también es paraphrase.

* STS-B: Benchmark de similiridad de oraciones sobre [STS-B](https://www.aclweb.org/anthology/S17-2001.pdf). Falta leer con más detenimiento.

Inferce tasks:

* MLI:

* QNLI: Se convierte el dataset SQuAD (que contiene pares de preguntas-contextos y sus respuestas como etiquetas) en un dataset de clasificación binaria (matched/mismatched) a paritir de todas las oraciones del contexto. Es decir, por cada oración del contexto se hace una muestra del nuevo dataset que determina si la respuesta a la pregunta está o no en la oración. Para hacer esto bien, hay que hacer varias cosas bastante elaboradas, pero hay trabajos hechos al respecto que sirven para varios idiomas (https://www.aclweb.org/anthology/I17-1100.pdf)

* RTE: Rejunte de datasets de *Recognizing Textual Entailment* (RTE1, RTE2, RTE3 y RTE5) que es lo mismo que NLI pero se juntan las categorías de neutral y not-entailed en una sola (not-entailed).

* WNLI: *Winograd Schema Challenge*. COnjunto de ejemplos que consisten en pronombres y frases y como etiquetas, el ente al que refiere el pronombre en la frase. Esto se convierte en una tarea de clasificación binaria reemplazando el pronombre por el ente en la frase y asignando la etiqueta de si corresponde o no con la frase original.



