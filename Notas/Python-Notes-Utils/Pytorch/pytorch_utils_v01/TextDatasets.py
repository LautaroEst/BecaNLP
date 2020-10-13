import torch
import torchtext
from torch.utils.data import DataLoader, sampler
from torchtext.datasets import text_classification


def GetAGNewsDataset(NUM_TRAIN, NUM_VAL, batch_size, n_grams):
    
    agnews_train, agnews_test = text_classification.DATASETS['AG_NEWS'](
        root='./Datasets/TextDatasets/AG_NEWS', ngrams=n_grams, vocab=None)
    
    DataDict = {}
    DataDict['n_train_samples'] = NUM_TRAIN # Cantidad de muestras para el conjunto de entrenamiento
    DataDict['n_val_samples'] = NUM_VAL # Cantidad de muestras para el conjunto de validación
    DataDict['n_test_samples'] = len(agnews_test) # Cantidad de muestras para el conjunto de testeo
    DataDict['batch_size'] = batch_size # Tamaño del batch
#     DataDict['size_of_one_input_sample'] = img.size() # Tamaño de cada muestra de entrada
#     DataDict['size_of_one_output_sample'] = 1 # Tamaño de cada muestra de salida
#     DataDict['output_categories'] = classes # Lista de categorías posibles a la salida
    
#     DataDict['train_dataloader'] = cifar10_train_dataloader # Dataloader de entrenamiento
#     DataDict['val_dataloader'] = cifar10_val_dataloader # Dataloader de validación
#     DataDict['test_dataloader'] = cifar10_test_dataloader # Dataloader de testeo
    print(agnews_train.get_vocab())


    return DataDict