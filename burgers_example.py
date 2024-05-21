# Python libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.modules):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, x):
        z = x
        for i in range(len(self.linears)-1):
            z = torch.tanh(self.linears[i](z))
        z = self.output(z)
        return z

def neural_net(layers):
    return MLP(layers)

