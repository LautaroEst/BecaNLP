import os
ROOT_PATH = os.path.join(__file__.split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/InterTASS2019task1')

import xml.etree.ElementTree as ET
import pandas as pd

# Dataset URUGUAY:
def uy_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_UY_train.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def uy_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_UY_dev.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def uy_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_UY_test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset Costa Rica:
def cr_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_CR_train.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def cr_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_CR_dev.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def cr_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_CR_test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset Perú:
def pe_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_PE_train.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def pe_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_PE_dev.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def pe_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_PE_test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset México:
def mx_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_MX_train.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def mx_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_MX_dev.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def mx_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_MX_test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset España:
def es_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_ES_train.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def es_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_ES_dev.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def es_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'TASS2019_country_ES_test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text



def train_reader(lang=['es']):
	
	reader_dict = {	'uy': uy_train_reader,
					'cr': cr_train_reader,
					'pe': pe_train_reader,
					'mx': mx_train_reader,
					'es': es_train_reader}

	if lang is None:
		lang = list(reader_dict.keys())

	for l in lang:
		reader = reader_dict[l]
		for text, label in reader():
			yield text, label


def dev_reader(lang=['es']):

	reader_dict = {	'uy': uy_dev_reader,
					'cr': cr_dev_reader,
					'pe': pe_dev_reader,
					'mx': mx_dev_reader,
					'es': es_dev_reader}

	if lang is None:
		lang = list(reader_dict.keys())

	for l in lang:
		reader = reader_dict[l]
		for text, label in reader():
			yield text, label


def test_reader(lang=['es']):

	reader_dict = {	'uy': uy_test_reader,
					'cr': cr_test_reader,
					'pe': pe_test_reader,
					'mx': mx_test_reader,
					'es': es_test_reader}

	if lang is None:
		lang = list(reader_dict.keys())

	for l in lang:
		reader = reader_dict[l]
		for tweet_id, text in reader():
			yield tweet_id, text


def get_train_dataframe(lang=['es']):
	tweets, labels = zip(*[(text,label) for text, label in train_reader(lang=lang)])
	return pd.DataFrame({'text':tweets, 'label':labels})


def get_dev_dataframe(lang=['es']):
	tweets, labels = zip(*[(text,label) for text, label in dev_reader(lang=lang)])
	return pd.DataFrame({'text':tweets, 'label':labels})


def get_test_dataframe(lang=['es']):
	tweet_ids, tweet_texts = zip(*[(tweet_id,text) for tweet_id, text in test_reader(lang=lang)])
	return pd.DataFrame({'tweet_id':tweet_ids, 'text':tweet_texts})