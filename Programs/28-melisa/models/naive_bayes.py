from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

import re
import string
non_ascii = 'áàâãäÁÀÂÃÄéèêëÉÈÊẼËíìîĩïÍÌÎĨÏóòôõöÓÒÔÕÖúùûũüÚÙÛŨÜñÑçÇ'
regex = r'[a-zA-Z{}]+|[{}]+|[{}]'.format(
                                            non_ascii,
                                            string.digits,
                                            re.escape(string.punctuation)
                                        )

import numpy as np

def naive_bayes_model(comments, labels):
	vec = CountVectorizer(token_pattern=regex,lowercase=True,
						ngram_range=(1,1),max_features=50000)
	clf = MultinomialNB()
	pipeline = Pipeline([
							('vect',vec),
							('clf', clf),
						])

	y_pred = cross_val_predict(pipeline, comments, labels, cv=5)
	print(confusion_matrix(labels,y_pred))