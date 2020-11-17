import torch
from torchvision import datasets as dset
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T


def GetCIFAR10DataLoaders(NUM_TRAIN, NUM_VAL, transform, batch_size=64):
    
#     transform = T.Compose([T.ToTensor(),
#                            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    transform = T.Compose([T.Resize(224),
                           T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    
    
    cifar10_train = dset.CIFAR10('CIFAR10/train/', train=True, download=True,
                                 transform=transform)
    cifar10_train_dataloader = DataLoader(cifar10_train, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('CIFAR10/train/', train=True, download=True,
                               transform=transform)
    cifar10_val_dataloader = DataLoader(cifar10_val, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN+NUM_VAL)))

    cifar10_test = dset.CIFAR10('CIFAR10/test/', train=False, download=True, 
                                transform=transform)
    cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=batch_size)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    img, label = cifar10_train[0]

    print()
    print('La base de datos CIFAR10 contiene {} muestras, cada una constituída por una imagen y la clase a la que pertenece.'.format(len(cifar10_train) + len(cifar10_test)))
    print('Cada muestra contiene dos componentes:')
    print('1) Input vector: imagen de tamaño {}x{}x{} con valores normalizados manualmente'.format(img.size()[0], img.size()[1], img.size()[2]))
    print('2) Label: ínidice de la lista de clases posibles: ')
    print(classes)
    print()
    print('Cantidad de muestras para entrenamiento: ', NUM_TRAIN)
    print('Cantidad de muestras para validación: ', NUM_VAL)
    print('Cantidad de muestras para test: ', len(cifar10_test))
       
    return cifar10_train_dataloader, cifar10_val_dataloader, cifar10_test_dataloader





class MNISTReshape(object):
    
    def __call__(self, sample):
        return sample[0,:,:]
        


def GetMNISTDataLoaders(NUM_TRAIN, NUM_VAL, batch_size=64):
    
    
    transform = T.Compose([T.ToTensor(),
                           MNISTReshape()])
    
    mnist_train_val = dset.MNIST('./MNIST/train', train=True, download=True, transform=transform)
    mnist_test = dset.MNIST('./MNIST/test', train=False, download=True, transform=transform)
    
    print('La base de datos MNIST contiene', len(mnist_train_val) + len(mnist_test), 'muestras')
    print('Cantidad de muestras para entrenamiento: ', NUM_TRAIN)
    print('Cantidad de muestras para validación: ', NUM_VAL)
    print('Cantidad de muestras para test: ', len(mnist_test))
    
    mnist_train_dataloader = DataLoader(mnist_train_val, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    mnist_val_dataloader = DataLoader(mnist_train_val, 
                                      batch_size=batch_size, 
                                      sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN,NUM_TRAIN+NUM_VAL)))
    mnist_test_dataloader = DataLoader(mnist_test, 
                                      batch_size=batch_size)  
    
    return mnist_train_dataloader, mnist_val_dataloader, mnist_test_dataloader