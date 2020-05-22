import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .BaseClassifiers import NeuralNetClassifier
from .BaseClassifiers import SequenceClassifier



class LogisticRegressionClassifier(NeuralNetClassifier):
    
    class Model(nn.Module):
        
        def __init__(self,in_features,bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features,1,bias)
            
        def forward(self,x):
            return self.linear(x)
    
    
    def __init__(self, in_features, bias=True, device='cpu'):
        model = self.Model(in_features, bias)
        super().__init__(model,device)
        
    def loss(self,scores,target):
        return F.binary_cross_entropy_with_logits(scores, target, reduction='mean')



class LinearSoftmaxClassifier(NeuralNetClassifier):

    class Model(nn.Module):
        
        def __init__(self,in_features,out_features,bias=True):
            super().__init__()
            self.linear = nn.Linear(in_features,out_features,bias)
            
        def forward(self,x):
            return self.linear(x)

    def __init__(self, in_features, n_classes, bias=True, device='cpu'):
        model = self.Model(in_features, n_classes, bias)
        super().__init__(model, device)
        
    def loss(self,scores,target):
        return F.cross_entropy(scores, target, reduction='mean')

    

class TwoLayerLRNet(NeuralNetClassifier):

    class Model(nn.Module):
        
        def __init__(self,in_features,hidden_features,bias=True,activation='relu'):
            super().__init__()
            self.linear1 = nn.Linear(in_features,hidden_features)
            self.linear2 = nn.Linear(hidden_features,1,bias)
            if activation == 'relu':
                self.act = F.relu
            elif activation == 'tanh':
                self.act = torch.tanh
            elif activation == 'sigmoid':
                self.act = torch.sigmoid
            else:
                raise TypeError('Activation function not supported')
            
        def forward(self,x):
            x = self.linear1(x)
            x = self.act(x)
            x = self.linear2(x)
            return x
    
    
    def __init__(self, in_features, hidden_features, bias=True, device='cpu'):
        model = self.Model(in_features, hidden_features, bias)
        super().__init__(model,device)
        
    def loss(self,scores,target):
        return F.binary_cross_entropy_with_logits(scores, target, reduction='mean')



class ManyToOneRecurrentClassifier(SequenceClassifier):
    """
    Implementación de un modelo end-to-end recurrente (Vanilla, LSTM o GRU).
    """
    class Model(nn.Module):

        def __init__(self,rnn,n_classes,*args,**kargs):
            if rnn == 'vanilla':
                self.rnn = nn.RNN(*args,**kwargs)
            elif rnn == 'lstm':
                self.rnn = nn.LSTM(*args,**kwargs)
            elif rnn == 'gru':
                self.rnn = nn.GRU(*args,**kwargs)
            else:
                raise NameError('Not supported {} recurrent model'.format(rnn))
            self.linear = nn.Linear(args[1],n_classes)

        def forward(x,x_lenghts):
            x = nn.utils.rnn.pack_padded_sequence(x,batch_first=True,lenghts=x_lenghts)
            x = self.rnn(x)
            x = self.linear(x.data)
            return x


    def __init__(self,device,rnn='vanilla',*args,**kwargs):
        model = self.Model(rnn,*args,**kwargs)
        super().__init__(model,device)

    def loss(self,scores,target):
        return F.cross_entropy(scores,target,reduce='mean')

    
    
    
    