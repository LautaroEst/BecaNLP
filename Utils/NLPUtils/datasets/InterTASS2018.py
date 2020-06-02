import os
ROOT_PATH = os.path.join(__file__.split('BecaNLP')[0],'BecaNLP')
DATASET_PATH = os.path.join(ROOT_PATH,'Utils/Datasets/InterTASS2018task1')

import xml.etree.ElementTree as ET
import pandas as pd

# Dataset Costa Rica:
def cr_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-CR-train-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def cr_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-CR-development-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def cr_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-CR-test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset Perú:
def pe_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-PE-train-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def pe_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-PE-development-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def pe_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-PE-test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


# Dataset España:
def es_train_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-ES-train-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def es_dev_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-ES-development-tagged.xml'))
	root = tree.getroot()
	for item in root:
		yield item[2].text, item[-1][0][0].text

def es_test_reader():
	tree = ET.parse(os.path.join(DATASET_PATH,'intertass-ES-test.xml'))
	root = tree.getroot()
	for item in root:
		yield item[0].text, item[2].text


def train_reader(lang=['es']):
	
	reader_dict = {	'cr': cr_train_reader,
					'pe': pe_train_reader,
					'es': es_train_reader}

	if lang is None:
		lang = list(reader_dict.keys())

	for l in lang:
		reader = reader_dict[l]
		for text, label in reader():
			yield text, label


def dev_reader(lang=['es']):

	reader_dict = {	'cr': cr_dev_reader,
					'pe': pe_dev_reader,
					'es': es_dev_reader}

	if lang is None:
		lang = list(reader_dict.keys())

	for l in lang:
		reader = reader_dict[l]
		for text, label in reader():
			yield text, label


def test_reader(lang=['es']):

	reader_dict = {	'cr': cr_test_reader,
					'pe': pe_test_reader,
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
