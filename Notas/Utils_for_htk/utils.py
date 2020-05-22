import re
import itertools

class Vocabulary(object):
    
    unk_token = '<UNK>'

    def __init__(self, tokens_dict=None, frequencies_dict=None):
        
        self._idx_to_tk = {} if tokens_dict is None else tokens_dict
        self._tk_to_idx = {tk: idx for idx, tk in tokens_dict.items()}
        self._idx_to_freq = {} if frequencies_dict is None else frequencies_dict
        self.max_idx = len(self)
        
    @classmethod
    def from_list_corpus(cls, corpus, cutoff_freq=1):
        corpus_words = sorted(list(set([tk for doc in corpus for tk in doc])))
        freqs_dict = {word: 0 for word in corpus_words}
        for tk in itertools.chain.from_iterable(corpus):
            freqs_dict[tk] += 1
        new_corpus_words, new_freqs_dict = {}, {}
        idx = 0
        for tk in corpus_words:
            if freqs_dict[tk] >= cutoff_freq:
                new_corpus_words[idx] = tk
                new_freqs_dict[idx] = freqs_dict[tk]
                idx += 1
        return cls(new_corpus_words, new_freqs_dict)

    def index_to_token(self, index):
        return self._idx_to_tk[index]

    def token_to_index(self, token):
        if token not in self._tk_to_idx.keys():
            return self.max_idx
        return self._tk_to_idx[token]

    def __repr__(self):
        return "<Vocabulary(size={})>".format(len(self))
    
    def __str__(self):
        text = ''
        for i, tk in enumerate(self):
            text += tk + '\n'
            if i > 10:
                text += '...'
                break
        return text

    def __len__(self):
        return len(self._idx_to_tk)
    
    def __getitem__(self,tk_or_idx):
        if isinstance(tk_or_idx, int):
            return self.index_to_token(tk_or_idx)
        if isinstance(tk_or_idx, str):
            return self.token_to_index(tk_or_idx)
        raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
    def __iter__(self):
        return (tk for tk in self._idx_to_tk.values())

    def __contains__(self,key):
        return key in self._idx_to_tk.values()
    
    def __add__(self,vocab):
        new_corpus_words = sorted(list(set(vocab._idx_to_tk.values()).union(set(self._idx_to_tk.values()))))
        new_idx_to_tk = {idx: tk for idx, tk in enumerate(new_corpus_words)}
        new_idx_to_freq = {idx: 0 for idx in range(len(new_corpus_words))}
        for idx, tk in enumerate(new_corpus_words):
            freq = self._idx_to_freq[self.token_to_index(tk)] if tk in self._idx_to_tk.values() else 0
            if tk in vocab._idx_to_tk.values():
                freq += vocab._idx_to_freq[vocab.token_to_index(tk)]
            new_idx_to_freq[idx] = freq
        return Vocabulary(new_idx_to_tk, new_idx_to_freq)
    

def read_binary_file(filename, decode='utf-8'):
    with open(filename, 'rb') as file:
        text = file.read().decode(decode)
    return text

def read_text_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def save_to_text_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

def save_to_binary_file(text, filename, encode='utf-8'):
    with open(filename, 'wb') as file:
        file.write(text.encode(encode))

def replace(text, pattern, repl):
    return re.sub(pattern, repl, text)

def insert_between(text, pat1, pat2, insert_expr):
    return re.sub(r'({})({})'.format(pat1,pat2), r'\g<1>{}\g<2>'.format(insert_expr), text)

def insert_right(text,pattern, insert_expr):
    return insert_between(text, pattern, '', insert_expr)

def insert_left(text, pattern, insert_expr):
    return insert_between(text, '', pattern, insert_expr)

def remove(text,pattern):
    return re.sub(pattern, '', text)

def split(text, delimiter):
    return re.split(delimiter,text)
    
    
    
class Corpus(object):
    
    def __init__(self, data, cutoff_freq=1):
        self.vocabulary = Vocabulary.from_list_corpus(data, cutoff_freq=cutoff_freq)
        self.docs_num = len(data)
        self.tokens_num = sum([len(doc) for doc in data])
        self.data = [[self.vocabulary.token_to_index(tk) for tk in doc] for doc in data]
        
        self.max_idx = self.tokens_num
    
    @classmethod
    def from_binary_files(cls, filenames, decode='utf-8', delimiter_pattern=' ', cutoff_freq=1):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'rb') as file:
                    texts_list.append(file.read().decode(decode))
        elif isinstance(filenames, str):
            with open(filenames, 'rb') as file:
                texts_list.append(file.read().decode(decode))
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_text_files(cls, filenames, delimiter_pattern=' ', cutoff_freq=1):
        texts_list = []
        if isinstance(filenames, list):
            for filename in filenames:
                with open(filename, 'r') as file:
                    texts_list.append(file.read())
        elif isinstance(filenames, str):
            with open(filenames, 'r') as file:
                texts_list.append(file.read())
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_strings(cls, texts, delimiter_pattern=' ', cutoff_freq=1):
        if isinstance(texts, list):
            texts_list = texts
        elif isinstance(filenames, str):
            texts_list = [texts]
        data = [re.split(delimiter_pattern, text) for text in texts_list]
        return cls(data, cutoff_freq=cutoff_freq)
    
    @classmethod
    def from_lists(cls, texts_list, cutoff_freq=1):
        return cls(texts_list, cutoff_freq=cutoff_freq)
    
    def __repr__(self):
        return "Corpus object\nNumber of docs = {}\nNumber of tokens = {}".format(self.docs_num, self.tokens_num)
    
    def __str__(self):
        printed_text = ''
        num_print_docs = min(self.docs_num,5)
        unk_token_idx = self.vocabulary.max_idx
        for i in range(num_print_docs):
            doc = self.data[i]
            if len(doc) <= 5:
                printed_text += repr([self.vocabulary.index_to_token(idx) if idx != unk_token_idx \
                                      else self.vocabulary.unk_token for idx in doc]) 
            else:
                printed_text += repr([self.vocabulary.index_to_token(idx) if idx != unk_token_idx \
                                      else self.vocabulary.unk_token for idx in doc[:4]])[:-1] + ', ...]'
            if i < num_print_docs:
                printed_text += '\n'
        if num_print_docs != self.docs_num:
            printed_text += '...'
        return printed_text

    def __len__(self):
        return self.tokens_num
    
    def __getitem__(self,tk_or_idx):
        if isinstance(tk_or_idx, int):
            return self.data[tk_or_idx]
        if isinstance(tk_or_idx, str):
            return [i for doc in self.data for i, tk in enumerate(doc)]
        raise KeyError('{} must be either integer or string'.format(tk_or_idx))
        
    def __iter__(self):
        return (self.vocabulary.index_to_token(idx) for doc in self.data for idx in doc)
    
    def __contains__(self,key):
        return key in self.vocabulary
                
#     def append(self,corpus):
#         if isinstance(corpus,Corpus):
#             self.vocabulary = self.vocabulary + corpus.vocabulary
#             self.docs_num += corpus.docs_num
#             self.tokens_num += corpus.tokens_num
#             idx_2_idx_dict = {idx: self.vocabulary._tk_to_idx[corpus.vocabulary._idx_to_tk[idx]] for idx in range(len(corpus.vocabulary))}
#             for doc in corpus.data:
#                 new_doc = [idx_2_idx_dict[idx] for idx in doc]
#                 self.data.append(new_doc)
#             self.max_idx = self.tokens_num
#         else:
#             raise TypeError('Sólo se puede anexar un corpus de tipo Corpus')