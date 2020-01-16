# Natural Language Processing para la Beca estímulo

---

## Antes del 10-09-2019

Hasta ahora vengo tratando de estudiar los temas de ML y de NLP por mi cuenta. Cuando me dí cuenta de que tenía que sentarme en serio empecé a ver un poco más de teoría y a tratar de hacer los ejercicios de Ng, pero después Leonardo me dijo lo de ver los datos de Twitter. 

## 10-09-2019

Hoy estuve buscando sentiment analysis de tweets. Hay muchos tutoriales que te enseñan a hacer cosas medio así nomás pero que están bastante interesantes (tipo Blobtext, Tweepy y otros). Después también encontré algunos papers un poco más trabajados.

## 13-09-2019

Estoy empezando a ver cómo distiguir tweets políticos. Primero se me ocurre hacer una representación semántica de los hashtags y con eso, sólo tendría que identificar qué hashtags son políticos y cuáles no. Es decir, cada # terndría asignado un word embedding aprendido en base a los twits en que aparece el #.

Ideas que aparecen en google:

* Cuando buscás si hay algo de esto hecho en google, aparece también la idea los datasets de word embeddings hechos a partir de twits.
Por ejemplo, [este](https://github.com/loretoparisi/word2vec-twitter).

* También encontré que se puede hacer el camino inverso: recomendar hashtags a los usuarios a partir de las cosas que twitean.

* Encontré un blog sobre el "día del panqueque" (en Argentina celebraríamos todos los cambios de gobierno), y lo que hace es buscar tweets un día en particular del año, para asegurarse de encontrar el significado correcto de ese hashtag.

* También se pueden usar hashtags como etiquetas y el tweet como input, y se aprende la representación de los features del texto.

* [Este](https://github.com/bdhingra/tweet2vec) creo que es lo que quiero hacer yo.

* [Este paper](https://www.ijcai.org/proceedings/2018/0480.pdf) tiene muy buena pinta. Dice que estudia los problemas de representar el word embedding de un hashtag.


**TO DO:**

1. Terminar el assignment 2 de cs224n.

2. Mirar el video de la [lecture 5](https://www.youtube.com/watch?v=nC9_RfjYwqA&feature=youtu.be) de dependency parsing

3. Empezar y terminar el assignment 3 de dependency parsing

4. No sé. Debería tratar de llegar a la lecture 11 y seguir desde ahí pero no creo que pueda hacerlo sin hacer primero el assignment 4 y mirar los videos de RNN y todos los que están en el medio.


## 15-09-2019

Domingo en lo de Viki. No hice lo que tenía que hacer del assignment 2. Voy a ver si puedo saltearlo y empezar a ver el video de de la lecture 5 (sí, tampoco vi la lecture 4 pq es básicamente de redes neuronales). El assignment 3 es de Pytorch.

Bueno, ahí vi el video de la lecture 5. No entendí mucho los algoritmos pero la idea es la siguiente. Una vez que se aprende el significado de las palabras, pasamos a aprender la estructura de la frase y las dependencias entre partes de ella... es decir, aprendemos a hacer análisis sintáctico automático. El video dice que hay un algoritmo determinístico que da buenos resultados (Nivre, 2003), pero que la cosa mejora cuando se usa ML y DL (sobre todo porque lo que hace mejorar la cosa es ponerle como input los word embeddings). El objetivo sería ahora hacer el assignment 3 para implementar algo de esto... y de paso aprender Pytorch.

Creo que acabo de entender algo importante (que no tiene que ver con la lecture 5). El orden de las cosas hasta ahora sería:

1. Aprendimos words embeddings por palabra. Es decir, tengo un texto y estimando probabilidades condicionales de palabras dado un contexto (o contexto dado palabra), conseguimos una representación vectorial de cada palabra. 

2. Una vez hecho esto, puedo usar esas representaciones para resolver tareas. En la lecture 3 se menciona la tarea de NER (name entity recognition), o sea, identificar como qué tipo de sustantivo se está usando la palabra (ej, Paris Francia, Paris Hilton, etc.). En la lecture 6, se usan los wordvectors para predecir texto (eso se llama "hacer un modelo de lenguaje").

3. Si yo quisiera usar wordembeddings para clasificar en positivo, negativo o neutra una oración, una primera idea sería promediar los words embeddings de cada palabra o de ventanas de palabras, pero no mantiene el orden de la frase. Por eso se usan las redes recurrentes, que manitenen un orden temporal. 

4. Una tarea similar que usa RNN porque quiere mantener ese orden temporal es la question answering o la de hacer resúmenes.

5. Cuando empieza el tema de RNN, empiezan a aparecer un montón de problemas que se resuelven con detalles de cálculos del gradiente y arquitecturas nuevas y complicadas.

6. En las lectures que siguen hace un paréntesis para la tarea de machine translation, que para eso se usa la arquitectura seq2seq y los mecanismos de atención. Éstos últimos también se usan para otras cosas, no sólo para traducción. 

7. Lo que entiendo es que en las tareas de machine translation y de QA los problemas a resolver son de arquitectura de la red.

8. También se pueden hacer varias de estas tareas con CNNs, aparentemente. En todos estos puntos no se hizo nada nuevo, más que cambiar la arquitectura. Es decir, lo que se viene haciendo es usar los wordembeddings y meterlos en una red (o en un modelo más simple) y resolver una tarea como NER, predición de contexto (modelo de lenguaje), QA, resumen o machine translation. 

9. La cosa empieza a cambiar un poco a partir de la lecture 12, que cuenta los modelos se sub-palabras. Después, la lecture 13 es de contextual representation y la lecture 14 es de Transformers y self-attention. Acá ya no se trabaja tanto sobre la arquitectura sino que se trata de hacer una variante sobre cómo representar el lenguaje. 

## 16-09-2019

Los lápices siguen escribiendo!

Estoy viendo la lecture 12, que habla de los subwords models. Es decir, hasta el momento se introducía a la red el wordembedding (o se aprendía), pero siempre a nivel palabra. Ahora la idea va a ser una representación con un vector, pero a nivel caracter o conjuntos de caracteres. Esto es bastante interesante, pero hasta ahora sólo lo está usando para MT. También se pueden encontrar words representations a partir de representaciones a nivel caracter. 

Lo que suele pasar es que a nivel caracter a veces no se captura el significado de la plabra. Ver LSTM-Character after gateway.

Acá muestran lo que hace [Fasttext](https://fasttext.cc/). VER!!!!

En la lecture 13 cuentan un poco más claramente la historia del entrenamiento con DL de NLP y cómo fue que el año pasado (2018) empezaron a usar otra representación de vectores: vectores contextualizados.


## 25-09-2019

Estos días estuve tratando de aprender a usar Pytorch. Voy a tratar de empezar a hacer los assignments del curso, a ver si avanzo más rápido.


## 1-10-2019

Resumen de las lectures:

* Lectures 1-4: Introducción a word embeddings y cómo definir y entrenar modelos.

* Lecture 5: Dependency parsing. Introducción a Pytorch.

* Lecture 6-8: Temas relacionados a RNN: básciamente, Language Model y Machine translation. 

Nota: Las RNN se usan para muchas tareas, no solo para predecir texto. (Es como si estuviera todo mezclado! Language model es parte de muchas tareas, las redes RNN se usan para un montón de cosas y esas otras cosas usan otras redes... qué complicado!)

* Lecture 9 y 10: Question Answering (y cosas del curso)

* Lecture 11: CNN. Empieza la segunda parte del curso.

* Lecture 12: Modelos de word vectors con subpalabras.

* Lecture 13: Context-representation of meanings.

Nota: Acá se dan cuenta que cuando arman la arquitectura de la red y después la entrenan, en realidad están haciendo una representación distribucional del significado de la palabra. Eso da origen a nuevos modelos.

Nota: La era pre-DL de NLP se basaba en extracción de features, que funcionaba mucho mejor que poner una red neuronal a aprender significados, o traducir, o responder preguntas. Cuando aparecen los wordvectors, lo que se hace es entrenar en forma no supervisada el significado de las palabras y después se busca aramar el modelo que realice la tarea. Eso es lo de la primera clase y lo que se venía haciendo hasta el 2018. En ese año empiezan a darse cuenta de que una arquitectura compleja que me resuelva una tarea, a veces me da un significado contextualizado de las palabras (eso se dieron cuenta cuando inicializaban aleatoriamente los word-vectors y los resultados subían muy poquito). Con esto, empiezan los modelos tipo TagLM, ELMO, BERT y los transformes que buscan representar no sólo el significado, sino también el registro, el contexto y la estructura gramatical, entre otras cosas. Esto caracteriza mucho mejor a la palabra y termina dando mejor resultado (a veces). Estos modelos se los conoce como Contextual Word Embeddings. Entonces, la idea de estos modelos sería tratar de reemplazar esa arquitectura que también termina por aprender (en forma distribuída) el significado contextualizado de la palabra por otro modelo que se concentre en hacer directamente eso. 

Nota: Una posible moraleja es que si tenés suficientes datos supervisados, los resultados no cambian mucho con word embeddings. 


## 2-10-2019

La cosa viene complicada. Hay MUCHO para estudiar y Leo quiere que ya empiece a hacer modelos sobre los tweets. Básicamente, el problemas es que hasta el 2018 se venían haciendo modelos con RNN y todas sus variantes y en ese año empezaron a hacer modelos con arquitectura de transformers que están dando muy buenos resultados. No entiendo muy bien qué es un transformer, pero me parece que todos estos modelos nuevos usan mezclas de los modelos viejos (o sea, "viejos" de antes del 2018) con esta cosa que se llama transformer. Nada es definitivo en este área pero igual habría que estudiarlos.


## 4-10-2019

Bueno, ayer hablé con Leo y después con mis viejos y estoy un poco más tranquilo. La verdad es que Leo tiene razón: primero hay que formarse bien, tapar la mayor cantidad de huecos posibles y después empezar a publicar. Me mandó a mirar un par de cosas (algunas medio que no muy urgentes) y después mi viejo me surgirió que mire otras cosas. 

También me doy cuenta que lo que yo quiero hacer en este momento es aprender learning theory (o lo que yo entiendo que es learning theory). O sea, concretamente, quiero entender bien cómo hacer un modelo de ML y qué implica teóricamente hacer ese modelo.

A partir de estos pensamientos, armo la lista de **TO DO**:

1. Curso de Ng de ML, la parte de Learning (correspondiente a las lectures 9, 10 y 11, y a las notas 4, 5 y 6)

2. Habiendo hecho esto, podría considerar leer los capítulos del Bishop, el capítulo 5 del Goodfellow y/o el curso de ML del McKay. Eso complementaría bastante lo del curso de Ng, pero también quitaría tiempo para otras cosas.

3. Después tendría que empezar a implementar algunos algoritmos. Pueden ser los del Bishop, los del curso de Ng de supervisado o algún otro (capaz hay material en el curso de cs230 o cs231). Ya podría mechar algún ejemplo de NLP acá.

4. Ahora sí, podría empezar a meterme con Deep Learning. Todavía no sé cómo sería bien esta parte, pero tengo varias opciones:

	1. El curso del francés.
	2. El curso de Ng (cs230)
	3. Leer el Goodfellow
	4. El curso de NLP.
	5. Combinaciones de lo anterior.

5. Quién sabe qué más...


## 5-10-2019

Vi los primeros dos videos de Learning del curso de Ng y tenía bastantes cosas que no sabía que existían. No era exactamente lo que esperaba (igual todavía me falta el último video), así que creo que vamos a darle una oportunidad de nuevo a los primeros 5 capítulos del Bishop y al capítulo 5 del Goodfellow.



## 22-10-2019


Contar lo que hice hasta ahora


**Nota importante**: El origen de las redes neuronales viene del hecho de que un sistema como el cerebro puede recibir las señales del exterior, separarlas en partes más simples y decidir en función de esas partes. Esta es la idea principal y más básica que toman los de computación para hacer un algoritmo que toma una decisión. 


## 6-11-2019

Hoy comienzo con las simulaciones. Estuve tratando de entender Pytorch pero hay varias cosas con las que tengo problemas:

1. Primero, que todo esto es negocio cuando tenés una GPU. Para bajos recursos esto no funciona muy eficientemente, con lo cual tampoco habría que matarse para hacer las cosas más eficientes.

2. Después, esto cambia muy rápido. No vale la pena hacer un paquete propio de utils o algo por el estilo porque en unos meses lo vas a tener que cambiar y no va a ser muy escalable. Lo que voy a empezar a hacer va a ser una serie de programas que cada uno vengan con sus funciones a parte (o en el mismo notebook o con una carpeta a parte) y que cumplan una función específica. Lo importante es la teoría.

Hay dos efoques a tomar, y ninguno de ellos incluye hacerse un código prolijito en Pytorch o en lo que sea que sirva para siempre: el primero es enfocarse en la teoría y el segundo, en la eficiencia (learning theory vs. sistemas embebidos). Ambos enfoques están buenos y se complementan pero los dos necesitan saber programar. Si voy a hacer teoría, tengo que tener recursos para testear las simulaciones. Pero, en este caso, la programación tiene que servir nada más para testear la teoría. No puedo perder tiempo en hacer cosas escalables o reutilizables a largo plazo y esas cosas que se hacen en embebidos. Por eso, me parece que la posta es hacer programas que cumplan una función específica y listo. 



## 8-11-2019

Ayer fui a verlo a Leo y me dijo que estaba muy bien lo que hice hasta ahora. Parece que la metodología "lento pero consistente" anda funcionando bien. 

Hasta ahora implementé Word2Vec en una manera muy rudimentaria, por lo que habría que mejorar eso. Por eso, el TO DO de este momento sería:

* Repasar teoría de Word2Vec, modelos de lenguaje, etc. y empezar a hacer un resumen sobre eso.

* Tratar de unir esa teoría con lo de antes (word embeddings, vector representation y todo eso). 

* Tratar de armase un paquete para probar los modelos que se van haciendo. Esto incluiría:

    * Los datasets más populares o los usados como estándar
    * Algunos datasets (estándares en la medida de lo posible)
    * Las formas de preprocesamiento posibles para levantar esos datasets (las más populares primero)
    * Las formas de medir los resultados (perplexity, analogía y si hay alguna otra forma, también).
    
* Obtener mejores resultados!! Supongo que voy a tener que leer muchos papers para esto...

## 15-01-2020 (Dos meses después...)

Holiiiiiiiis!! NLP De nuevo en acción. Rendí economía y volví de vacaciones, así que ya puedo ponerme de vuelta con esto. 

Estuve probando varias cosas en el medio que no anoté, pero ya tengo más o menos armadas algunas funciones que no sé si funcionan porque no puedo visualizar bien los resultados. La idea, después de hablar con mis viejos un poco, va a ser encontrar un paper baseline que me sirva para comparar mis resultados. Por ahora encontré el de [Baroni et al. 2014](https://www.aclweb.org/anthology/P14-1023/) que parece ser el más conveniente, aunque bastante viejo. Vamos a tratar de hacer eso y después volvemos.


