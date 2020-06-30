import nltk
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer, word_tokenize
import re

tweet_tknzr = TweetTokenizer().tokenize

def tokenize_tweet(tweet):
    return tweet_tknzr(tweet)

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def tokenize_characters(string):
	return list(string)

def tokenize_from_pattern(pattern,string):
	return re.findall(pattern,string)

