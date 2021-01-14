

- Diferentes tipos de sentiment analysis:

	* Document-level classification. Clasificar cada comentario por separado en pos/neg o 1-5. Esto se puede hacer en categorías por separado y estudiar las correlaciones. Pensar en correlaciones entre usuarios. 

	* Sentence-level classification. Clasificar oraciones en objetivas y subjetivas. Puede que haya una correlación en dónde se encuentran las oraciones objteivas y las sibjetivas, si es en los comentarios con más rate o menos rate. Puede que la cantidad de likes esté relacionada con la objetividad de las oraciones.

	* Aspect-based. Esta es más difícil y habría que pensarla un poco.

	* Multi-domain analysis. Esto trata de usar información en un tema y trasladarla a otro. Supongo que se refiere a los productos, por ejemplo. 

	* Multimodal: sentimientos de varias fuentes, no solo texto.

- Ver análisis hecho con un corpus similar en [este](./Fang-Zhan2015_Article_SentimentAnalysisUsingProductR.pdf) paper.

# Principales problemas a resolver  con MeLiSA

- Si todavía no se tiene armado el corpus, lo que se me ocurre que es válido es hacer k-fold cross-validation con todo el dataset para varios modelos. Esto me da acceso a k matrices de confusión por modelo. Una primera opción para ver el dataset es analizar si hay patrones en estas matrices de confusión. Desventajas: Es bastante costoso este procedimiento y no sé si tiene resultados directos. Mi hipótesis es que si hay muchos más comentarios positivos que negativos entonces debería haber un sesgo a los positivos. ¿Qué pasa si modifico la proporción de positivos y negativos? ¿Cambia en algo? Esto podría hacerse para fine-grained y para pos/neg. También se puede pensar en algo parecido con el largo de los comentarios. ¿Hay un sesgo en esa variable también?

- Uno de los problemas que tiene SA es que es difícil clasificar mirando palabras y ocurrencias de palabras como se hace en clasificación de documentos. Por eso, estaría bien incluír en MeLiSA los ejemplos que tienen palabras positivas pero son negativos y viceversa. **Esto tiene que ser cabecera: identificar los ejemplos que sí o sí tienen que estar en MeLiSA**. Una forma de conseguir esto se me ocurre que puede ser usando un clasificador e identificando los ejemplos mal clasificados e investigar dónde falla el modelo. Para esto es necesario ya tener armado el corpus. 

