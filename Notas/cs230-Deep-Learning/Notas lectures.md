# Notas de las lectures del cuerso

Lectures para ver todas seguidas que son sobre cosas de deep learning y proyectos en genera:

* Lecture 1 (Introducción)
* Lecture 2 (Intuición de deep learning)
* Lecture 3 (Proyectos en deep learning)
* Lecture 6 (Estrategias para proyectos de deep learning)
* Lecture 8 (Cómo leer papers)
* Lecture 10 (Conclusiones)

El curso 1 es bastante práctico y no dice mucho sobre las cuestiones teóricas de deep learning. Básicamente te explica cómo programar una red deep feed forward con numpy y cómo hacer las cuentas. 

Los cursos 2 y 3 (sobre todo el 2) te dice los trucos con los cuales funcionan las redes neuronales así que hay que prestarles bastante atención.

## Curso 2

* El curso empieza explicando la diferencia entre bias y variance y error bayesiano. Esto hay que agregarlo a las notas de learning.

* También hace muchas idas y vueltas entre cómo diseñar algoritmos de ML. En el segundo video, dice los pasos a seguir para sacarse de encima el bias y la varianza y dice que el trade off entre esas dos cosas no debería existir en DL. 

```
	    <------------------------------------------------------
	    |                                                     |
	tengo bias? ---Y---> busco un modelo más grande           |
	    |                entrenar por más tiempo la red  ---->|
	    N                                                     |
	    |                                                     |
	tengo variance? ---Y---> busco más datos                  |
	    |                    Regularization   --------------->|
	    N
	    |
	 Terminé
	 
```
Esto va en la parte de learning theory del resumen de supervisado.

* Después habla de regularización: L2, L1, dropout, early stopping. Esto es importante, y hay que implementarlo.

* Después empieza con todas las cosas que necesitamos para optimizar una función: inicialización de los parámetros, normalización de las entradas y gradient checking





