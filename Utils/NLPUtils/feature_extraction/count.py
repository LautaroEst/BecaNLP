from sklearn.feature_extraction.text import CountVectorizer as skcv
from collections import Counter, defaultdict
from itertools import tee, islice
from scipy.sparse import csr_matrix
from tqdm import tqdm

class CorpusCountVectorizer(skcv):
	pass


def get_ngrams(doc, ngram_range=(1,1)):

	for n in range(ngram_range[0],ngram_range[1]+1):
	    tlst = doc
	    while True:
	        a, b = tee(tlst)
	        l = tuple(islice(a, n))
	        if len(l) == n:
	            yield ' '.join(l)
	            next(b)
	            tlst = b
	        else:
	            break


def count_bag_of_ngrams(corpus, ngram_range=(1,1), tokenizer=None):
	
	if tokenizer is None:
		tokenizer = lambda x: x

	data = []
	indices = []
	indptr = [0]

	full_vocab = defaultdict()
	full_vocab.default_factory = full_vocab.__len__

	for doc in tqdm(corpus):
		features = dict(Counter(get_ngrams(tokenizer(doc),ngram_range)))
		data.extend(features.values())
		indices.extend([full_vocab[tk] for tk in features])
		indptr.append(len(indices))

	vocab_len = len(full_vocab)
	X = csr_matrix((data,indices,indptr),shape=(len(corpus),vocab_len))
	return X, dict(full_vocab)




