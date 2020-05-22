import torch
import re
import nltk
import itertools

nltk.download('brown', download_dir='/home/lestien/anaconda3/envs/TorchEnv/nltk_data')
from nltk.corpus import brown

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader



class BrownDataset(torch.utils.data.Dataset):
    
    def __init__(self, categories, root='./', preprocessing=None, context_size=2):
        nltk.download('brown', download_dir=root)
        from nltk.corpus import brown
        self.corpus_unpreproceced = brown.sents(categories=categories)
        self.preprocessing = preprocessing
        self.context_size = context_size
        
        if self.preprocessing:
            self.corpus = self.preprocessing(self.corpus_unpreproceced)
        else:
            self.corpus = self.corpus_unpreproceced
        
        no_sentence = '<NS>'
        self.vocabulary = set(itertools.chain.from_iterable(self.corpus))
        self.vocabulary.add(no_sentence)

        self.word_to_index = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(self.vocabulary)}
        
        samples = []
        for sentence in self.corpus:
            for i, word in enumerate(sentence):
                first_context_word_index = max(0,i-self.context_size)
                last_context_word_index = min(i+self.context_size+1, len(sentence))
                
                context = [no_sentence for j in range(i-self.context_size,first_context_word_index)] + \
                          sentence[first_context_word_index:i] + \
                          sentence[i+1:last_context_word_index] + \
                          [no_sentence for j in range(last_context_word_index,i+self.context_size+1)]
                
                samples.append((context, word))
        
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        
        context, word = self.samples[idx]
        idx_context = torch.empty(len(context), dtype=torch.long)
        idx_word = torch.tensor(self.word_to_index[word], dtype=torch.long)
        for i, w in enumerate(context):
            idx_context[i] = self.word_to_index[w]

        return idx_context, idx_word
       
        
class PreprocessBrown(object):
    
    def __call__(self,corpus_unpreproceced):
        corpus = []
        for sentence in corpus_unpreproceced:
            text = ' '.join(sentence)
            text = text.lower()
            text.replace('\n', ' ')
            text = re.sub('[^a-z ]+', '', text)
            corpus.append([w for w in text.split() if w != ''])
        return corpus




class Word2VecCBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecCBOW,self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, context_word):
        emb = self.embeddings(context_word).mean(dim=1)
        return self.linear(emb)
    
    def loss(self, scores, target):
        m = nn.CrossEntropyLoss()
        return m(scores,target)
    




def TrainWord2Vec(model, data, epochs=1, learning_rate=1e-2, sample_loss_every=100, lm='CBOW'):
    
    categories = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', \
                  'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', \
                  'reviews', 'romance', 'science_fiction']
    context_size = 2
    train_dataset = BrownDataset(categories=categories,
                                 root='/home/lestien/anaconda3/envs/TorchEnv/nltk_data',
                                 preprocessing=PreprocessBrown(),
                                 context_size=context_size)

    val_dataset = BrownDataset(categories=categories,
                                 root='/home/lestien/anaconda3/envs/TorchEnv/nltk_data',
                                 preprocessing=PreprocessBrown(),
                                 context_size=context_size)

    batch_size = 64
    NUM_TRAIN = len(train_dataset)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  sampler=SubsetRandomSampler(range(NUM_TRAIN)))

    vocab_size = len(train_dataset.vocabulary)
    embedding_size = 50
    model = Word2VecCBOW(vocab_size, embedding_size)
    
    
    input_dtype = data['input_dtype'] 
    target_dtype = data['target_dtype']
    device = data['device']
    train_dataloader = data['train_dataloader']
    val_dataloader = data['val_dataloader']
    
    performance_history = {'iter': [], 'loss': [], 'accuracy': []}
    
    model.train()
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        for t, (x,y) in enumerate(train_dataloader):
            x = x.to(device=device, dtype=input_dtype)
            y = y.to(device=device, dtype=target_dtype)

            if lm == 'CBOW':
                scores = model(x) # Forward pass
                loss = model.loss(scores,y) # Backward pass
                
            elif lm == 'SkipGram':
                y = y.view(-1,1)
                y = y.expand(-1,x.size()[1])
                scores = model(y)
                loss = model.loss(scores,x)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % sample_loss_every == 0:
                print('Epoch: %d, Iteration: %d, Loss: %d/%d ' % (e, t, loss.item()))
                
    return model.embeddings
