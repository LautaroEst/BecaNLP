import pandas as pd
import numpy as np
import spacy
from spacy_langdetect import LanguageDetector
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import langid
from langid.langid import LanguageIdentifier, model
import fasttext
import time

df_esp = pd.read_csv('./reviews_esp_cleaned_chars_11-01-2021.csv')
df_por = pd.read_csv('./reviews_por_cleaned_chars_11-01-2021.csv')
df = pd.concat([df_esp,df_por],ignore_index=True)
ds_text = (df['review_content'] + ' ' + df['review_title']).astype(np.str)

threshold = .9

def detect_lang_spacy_es(ds):
	nlp = spacy.load('es_core_news_sm')
	nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
	lang_score = pd.DataFrame([doc._.language for doc in nlp.pipe(ds)])
	true_results = lang_score['score'] > threshold
	por_results = (lang_score['language'] == 'pt') & true_results & (df['country'] == 'MLB') 
	esp_results = (lang_score['language'] == 'es') & true_results & (df['country'] != 'MLB') 
	return por_results | esp_results


def detect_lang_spacy_pt(ds):
	nlp = spacy.load('pt_core_news_sm')
	nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
	lang_score = pd.DataFrame([doc._.language for doc in nlp.pipe(ds)])
	true_results = lang_score['score'] > threshold
	por_results = (lang_score['language'] == 'pt') & true_results & (df['country'] == 'MLB')
	esp_results = (lang_score['language'] == 'es') & true_results & (df['country'] != 'MLB') 
	return por_results | esp_results


def detect_lang_langdetect(ds):

	def detect_language(text):
		try:
			lang_score = detect_langs(text)[0].__dict__
		except LangDetectException:
			lang_score = {'lang':'es', 'prob': 0.}
		return lang_score

	lang_score = pd.DataFrame(ds.apply(detect_language).tolist())
	true_results = lang_score['prob'] > threshold
	por_results = (lang_score['lang'] == 'pt') & true_results & (df['country'] == 'MLB') 
	esp_results = (lang_score['lang'] == 'es') & true_results & (df['country'] != 'MLB') 
	return por_results | esp_results

def detect_lang_langid(ds):
	lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

	def apply_lang_detect(text):
		return dict(zip(*[('lang','prob'),lang_identifier.classify(text)]))

	lang_score = pd.DataFrame(ds.apply(apply_lang_detect).tolist())
	true_results = lang_score['prob'] > threshold
	por_results = (lang_score['lang'] == 'pt') & true_results & (df['country'] == 'MLB') 
	esp_results = (lang_score['lang'] == 'es') & true_results & (df['country'] != 'MLB') 
	return por_results | esp_results

def detect_lang_fasttext(ds):
	model_predict = fasttext.load_model('../27-mercado-libre-api-v2/lid.176.bin').predict

	def apply_lang_detect(text):
		return dict(zip(*[('lang','prob'),next(zip(*model_predict(text, k=1)))]))

	lang_score = pd.DataFrame(ds.apply(apply_lang_detect).tolist())
	true_results = lang_score['prob'] > threshold
	por_results = (lang_score['lang'] == '__label__pt') & true_results & (df['country'] == 'MLB') 
	esp_results = (lang_score['lang'] == '__label__es') & true_results & (df['country'] != 'MLB') 
	return por_results | esp_results


if __name__ == "__main__":
	tic = time.time()
	mask_spacy_es = detect_lang_spacy_es(ds_text)
	toc = time.time()
	print('Spacy-es:',toc-tic)
	mask_spacy_es.to_csv('./mask_spacy_es.csv',index=False)

	tic = time.time()
	mask_spacy_pt = detect_lang_spacy_pt(ds_text)
	toc = time.time()
	print('Spacy-pt:',toc-tic)
	mask_spacy_es.to_csv('./mask_spacy_pt.csv',index=False)

	tic = time.time()
	mask_langdetect = detect_lang_langdetect(ds_text)
	toc = time.time()
	print('langdetect:',toc-tic)
	mask_langdetect.to_csv('./mask_langdetect.csv',index=False)

	tic = time.time()
	mask_langid = detect_lang_langid(ds_text)
	toc = time.time()
	print('langid:',toc-tic)
	mask_langid.to_csv('./mask_langid.csv',index=False)

	tic = time.time()
	mask_fasttext = detect_lang_fasttext(ds_text)
	toc = time.time()
	print('fasttext:',toc-tic)
	mask_fasttext.to_csv('./mask_fasttext.csv',index=False)

