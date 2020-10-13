import torch
import torchtext
from torch.utils.data import DataLoader, sampler
from torchtext.datasets import text_classification


def GetAGNewsDataset(val_size, batch_size, n_grams):
    
    agnews_train, agnews_test = text_classification.DATASETS['AG_NEWS'](
        root='./Datasets/TextDatasets/AG_NEWS', ngrams=n_grams, vocab=None)
    
    NUM_TRAIN = int((1 - val_size) * len(agnews_train))
    NUM_VAL = len(agnews_train) - NUM_TRAIN

    agnews_train_dataloader = DataLoader(agnews_train, 
                                    batch_size=batch_size, 
                                    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    agnews_val_dataloader = DataLoader(agnews_train, 
                                        batch_size=batch_size, 
                                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN+NUM_VAL)))
    
    agnews_test_dataloader = DataLoader(agnews_test, batch_size=batch_size)
    
    return agnews_train_dataloader, agnews_val_dataloader, agnews_test_dataloader
