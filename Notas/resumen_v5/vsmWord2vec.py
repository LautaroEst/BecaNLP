from vsmUtils import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import argparse


class SkipGramModel(nn.Module):
    
    def __init__(self,embedding_dim,vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1,embedding_dim,padding_idx=vocab_size)
        self.out = nn.Linear(embedding_dim,vocab_size,bias=False)
    
    def forward(self,x):
        return self.out(self.emb(x))
    

def Loss(scores,y,ignore_idx):
    return nn.functional.cross_entropy(scores,y,ignore_index=ignore_idx)


def train(dataloader,
          embedding_dim=5,
          vocab_size=10,
          epochs=1,
          learning_rate=1e-3):
    
    try:
        print('Starting training...')
        device = torch.device('cpu')
        model = SkipGramModel(embedding_dim,vocab_size).to(device=device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        for e in range(epochs):
            for t, (x,y) in enumerate(dataloader):
                #x = x.to(device=device)
                #y = y.to(device=device)

                optimizer.zero_grad() # Llevo a cero los gradientes de la red
                scores = model(x) # Calculo la salida de la red
                loss = Loss(scores,y,vocab_size) # Calculo el valor de la loss
                loss.backward() # Calculo los gradientes
                optimizer.step() # Actualizo los parámetros
                
    except KeyboardInterrupt:
        print('Exiting training...')

        
def test_word2vec():
    ROOT_PATH = '../../Utils/Datasets/aclImdb/train/unsup/'
    split_fn = lambda x: x.split(' ')
    corpus_idx, idx_to_tk = vsmUtils.get_corpus_and_vocab(ROOT_PATH, split_fn)
    
    vocab_size = len(idx_to_tk)
    left_n, right_n = 2, 1
    dataset = WindowTextDataset(corpus_idx,left_n,right_n, unk_idx=vocab_size)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    embedding_dim = 5
    learning_rate = 1e-3
    epochs = 1
    train_skipgram(dataloader,embedding_dim,vocab_size,epochs,learning_rate)
        

if __name__ == '__main__':
    d = parse_args()
    test_word2vec()
    