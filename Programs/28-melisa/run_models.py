import models as m
import pandas as pd
import numpy as np
import pathlib

df = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + '/data_esp_sample.csv')
comments = df['review_content'].values.astype(np.str)
labels = df['review_rate'].values.astype(np.int)

if __name__ == "__main__":
	m.xlm_model()
	m.multibert_model()
	m.beto_model()
	m.rnn_model()
	m.naive_bayes_model(comments, labels)
