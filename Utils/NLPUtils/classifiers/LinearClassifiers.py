import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NeuralNetClassifier



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
    