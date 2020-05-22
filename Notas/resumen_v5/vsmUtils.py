import os
import torch
from torch.utils.data import Dataset

def get_corpus_and_vocab(root, split_fn):
    """
    Función para obtener el corpus de un conjunto de archivos
    de texto ubicados en la carpeta root.
    
    Argumentos:
        * root: carpeta donde se encuentran los archivos
        que contienen el corpus.
        * split_fn: función para separar el texto de los 
        archivos.
    
    Devuelve:
        * corpus_idx: Una lista de listas de enteros que
        contiene los índices de las palabras del corpus.
        * idx_to_tk: Diccionario con los índices de las
        palabras encontradas en el corpus.
    """
    filenames = os.listdir(root)[:3]
    corpus_idx = []
    tk_to_idx = {}
    vocab_len = 0
    for filename in filenames:
        with open(os.path.join(root,filename), 'r') as f:
            doc = split_fn(f.read())
        corpus_idx.append([])
        doc_idx = corpus_idx[-1]
        for tk in doc:
            if tk not in tk_to_idx:
                tk_to_idx[tk] = vocab_len
                vocab_len += 1
            doc_idx.append(tk_to_idx[tk])
            
    idx_to_tk = {idx: tk for tk, idx in tk_to_idx.items()}
    return corpus_idx, idx_to_tk    


class ContextWindowDataset(Dataset):
    
    """
    Muestras de tipo "contexto - palabra central" obtenidas de un 
    corpus de texto con una ventana fija.
    
    Argumentos:
        * corpus_idx: lista de lista de enteros que contienen los 
        índices de cada palabra del vocabulario.
        * left_n / right_n: Cantidad de palabras a la izquierda /
        derecha de la palabra central.
        * unk_idx: valor del ínidice a ser tomado por el modelo
        como índice desconocido.
        
    Devuelve:
        * __getitem__(self,idx): devuelve la muestra idx del dataset.
        * __len__(self): devuelve la cantidad de muestras del dataset.
    """
    
    def __init__(self,corpus_idx,left_n,right_n, unk_idx):
        self.words, self.contexts = self.get_samples(corpus_idx,left_n,right_n, unk_idx)
        
    def __getitem__(self,idx):
        return self.words[idx], self.contexts[idx]
    
    def __len__(self):
        return len(self.words)
    
    def get_samples(self, corpus_idx, left_n, right_n, unk_idx=-1):
        unk_token_idx = unk_idx
        context_size = left_n + right_n
        words = []
        contexts = []
        for doc in corpus_idx:
            for i in range(left_n):
                doc.insert(0,unk_token_idx)
            for i in range(right_n):
                doc.append(unk_token_idx)
            for i, idx in enumerate(doc[left_n:-right_n],left_n):
                words.append(idx)
                contexts.append(doc[i-left_n:i] + doc[i+1:i+right_n+1])
            for i in range(left_n):
                doc.pop(0)
            for i in range(right_n):
                doc.pop(-1)
        words = torch.tensor(words).view(-1,1).repeat(1,context_size).view(-1)
        contexts = torch.tensor(contexts).view(-1)
        mask = (words != unk_token_idx) * (contexts != unk_token_idx)
        return words[mask], contexts[mask]
    
    