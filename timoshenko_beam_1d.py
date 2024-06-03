# Python libs
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Defining model objects
@dataclass
class TimoshenkoBeam:
    length: float
    G: float
    E: float
    As: float
    I: float

@dataclass
class BoundaryConditions:
    x_BC_1: np.ndarray
    w_train: np.ndarray
    psi_train: np.ndarray
    x_BC_2: np.ndarray
    psi_x_train: np.ndarray
    psi_xx_train: np.ndarray
    w_pred: np.ndarray
    psi_pred: np.ndarray
    psi_x_pred: np.ndarray
    psi_xx_pred: np.ndarray

class MLP(nn.Module):
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
    
def w_NN(layers):
    return MLP(layers)

def psi_NN(layers):
    return MLP(layers)

def data_loss(bc: BoundaryConditions):
    return torch.mean((bc.w_train - bc.w_pred)**2) + torch.mean((bc.psi_train - bc.psi_pred)**2) + torch.mean((bc.psi_x_train - bc.psi_x_pred)**2) + torch.mean((bc.psi_xx_train - bc.psi_xx_pred)**2)

def physical_loss(model, X_f_train, psi_x_train, psi_xx_train, G, E, As, I):
    pass


def loss_function(model, X_u_train, w_train, psi_train, X_f_train, psi_x_train, psi_xx_train, G, E, As, I):
    pass


if __name__ == '__main__':
    # Parameters
    length = 100
    G = 5e6
    E = 1e7
    As = 1
    I = 1/12
    F = 480

    # Hyperparameters
    lr = 0.01
    epochs = 1000
    layers = [2, 50, 50, 50]

    # Training data
    N_u = 100
    N_f = 10000

    # Boundary conditions: w(0) = 0, psi(0) = 0
    x_BC_1 = np.zeros((N_u, 1))
    w_train = np.zeros((N_u, 1))
    psi_train = np.zeros((N_u, 1))

    # Boundary condtions: psi_x(L) = 0, psi_xx(L) = F
    x_BC_2 = np.ones((N_u, 1))*length
    psi_x_train = np.zeros((N_u, 1))
    psi_xx_train = np.ones((N_u, 1))*F/(E*I)
    
    # Collocation points
    X_f_train = np.random.uniform(0, length, (N_f, 1))
