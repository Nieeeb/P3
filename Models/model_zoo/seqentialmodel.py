from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F

class Sequential(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module):
        self.model1 = model1
        self.model2 = model2        


    def forward(self,x):
        x1 = self.model1.forward(x)
        x2 = self.model2.forward(x1)

        return x2