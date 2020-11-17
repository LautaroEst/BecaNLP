import torch
from torchvision import datasets as dset
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T


def GetCIFAR10Dataset(NUM_TRAIN, NUM_VAL, transform, batch_size=64):
    
    # Se descarga el dataset y se definen los dataloaders de train / validation / test:
    cifar10_train = dset.CIFAR10('Datasets/TorchvisionDatasets/CIFAR10/train', train=True, download=True,
                                 transform=transform)
    cifar10_train_dataloader = DataLoader(cifar10_train, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('Datasets/TorchvisionDatasets/CIFAR10/train', train=True, download=True,
                               transform=transform)
    cifar10_val_dataloader = DataLoader(cifar10_val, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN+NUM_VAL)))

    cifar10_test = dset.CIFAR10('Datasets/TorchvisionDatasets/CIFAR10/test/', train=False, download=True, 
                                transform=transform)
    cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=batch_size)
    
    
    # Se guarda toda la información en un diccionario:
    DataDict = {}
    
    img, label = cifar10_train[0]
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    DataDict['n_train_samples'] = NUM_TRAIN # Cantidad de muestras para el conjunto de entrenamiento
    DataDict['n_val_samples'] = NUM_VAL # Cantidad de muestras para el conjunto de validación
    DataDict['n_test_samples'] = len(cifar10_test) # Cantidad de muestras para el conjunto de testeo
    DataDict['batch_size'] = batch_size # Tamaño del batch
    DataDict['size_of_one_input_sample'] = img.size() # Tamaño de cada muestra de entrada
    DataDict['size_of_one_output_sample'] = 1 # Tamaño de cada muestra de salida
    DataDict['output_categories'] = classes # Lista de categorías posibles a la salida
    
    DataDict['train_dataloader'] = cifar10_train_dataloader # Dataloader de entrenamiento
    DataDict['val_dataloader'] = cifar10_val_dataloader # Dataloader de validación
    DataDict['test_dataloader'] = cifar10_test_dataloader # Dataloader de testeo
       
    
    return DataDict

       


def GetMNISTDataset(NUM_TRAIN, NUM_VAL, transform, batch_size=64):
    
    # Se descarga el dataset y se definen los dataloaders de train / validation / test:
    mnist_train = dset.MNIST('Datasets/TorchvisionDatasets/MNIST/train', train=True, download=True,
                                 transform=transform)
    mnist_train_dataloader = DataLoader(mnist_train, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    mnist_val = dset.MNIST('Datasets/TorchvisionDatasets/MNIST/train', train=True, download=True,
                               transform=transform)
    mnist_val_dataloader = DataLoader(mnist_val, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN+NUM_VAL)))

    mnist_test = dset.MNIST('Datasets/TorchvisionDatasets/MNIST/test/', train=False, download=True, 
                                transform=transform)
    mnist_test_dataloader = DataLoader(mnist_test, batch_size=batch_size)
    
    
    # Se guarda toda la información en un diccionario:
    DataDict = {}
    
    img, label = mnist_train[0]
    classes = list(range(10))
    DataDict['n_train_samples'] = NUM_TRAIN # Cantidad de muestras para el conjunto de entrenamiento
    DataDict['n_val_samples'] = NUM_VAL # Cantidad de muestras para el conjunto de validación
    DataDict['n_test_samples'] = len(mnist_test) # Cantidad de muestras para el conjunto de testeo
    DataDict['batch_size'] = batch_size # Tamaño del batch
    DataDict['size_of_one_input_sample'] = img.size() # Tamaño de cada muestra de entrada
    DataDict['size_of_one_output_sample'] = 1 # Tamaño de cada muestra de salida
    DataDict['output_categories'] = classes # Lista de categorías posibles a la salida
    
    DataDict['train_dataloader'] = mnist_train_dataloader # Dataloader de entrenamiento
    DataDict['val_dataloader'] = mnist_val_dataloader # Dataloader de validación
    DataDict['test_dataloader'] = mnist_test_dataloader # Dataloader de testeo
    
    return DataDict