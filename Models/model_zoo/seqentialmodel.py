from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F

class SequentialModel(nn.Module):
    def __init__(self, model1, model2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model1 = model1
        self.model2 = model2 


    def forward(self,x):
        x1 = self.model1.forward(x)
        x2 = self.model2.forward(x1)

        return x2