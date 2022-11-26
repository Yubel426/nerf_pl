import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()

    def forward(self,x):
        return x
    