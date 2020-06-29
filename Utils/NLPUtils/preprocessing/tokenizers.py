import nltk
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer, word_tokenize

tweet_tknzr = TweetTokenizer().tokenize

def tokenize_tweet(tweet):
    return tweet_tknzr(tweet)

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def tokenize_characters(string):
	return list(string)
