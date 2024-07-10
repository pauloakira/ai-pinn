# Python libs
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Custom libs
from domain.models import TimoshenkoBeam, TimoshenkoBoundary
from utils.mlflow_helper import mlflowPipeline


# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Setting up Metal
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, layers: List[int]):
        super(MLP, self).__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, x):
        z = x
        for i in range(len(self.linears)-1):
            z = torch.tanh(self.linears[i](z))
        z = self.linears[-1](z)
        return z


def w_NN(layers: List[int]) -> MLP:
    """
    Creates and returns a neural network for the deflection (w) field using the provided layer configuration.

    Args:
        layers (list): A list specifying the number of neurons in each layer of the neural network.

    Returns:
        MLP: An instance of the MLP class representing the neural network for the deflection field.
    """
    return MLP(layers).to(device)

def psi_NN(layers: List[int]) -> MLP:
    """
    Creates and returns a neural network for the rotation (psi) field using the provided layer configuration.

    Args:
        layers (list): A list specifying the number of neurons in each layer of the neural network.

    Returns:
        MLP: An instance of the MLP class representing the neural network for the rotation field.
    """
    return MLP(layers).to(device)

def data_loss(bc: TimoshenkoBoundary, bc_pred: TimoshenkoBoundary)->torch.Tensor:
    """
    Computes the data loss for the deflection (w) and rotation (psi) fields.

    Args:
        bc (TimoshenkoBoundary): An instance of the TimoshenkoBoundary class containing the true boundary conditions.
        bc_pred (TimoshenkoBoundary): An instance of the TimoshenkoBoundary class containing the predicted boundary conditions.

    Returns:
        torch.Tensor: The data loss for the deflection and rotation fields.
    """
    return torch.mean((bc.w_0 - bc_pred.w_0)**2) + torch.mean((bc.w_L - bc_pred.w_L)**2) + torch.mean((bc.psi_0 - bc_pred.psi_0)**2) + torch.mean((bc.psi_L - bc_pred.psi_L)**2)

def physical_loss(w_model: MLP, psi_model: MLP, X_f_train: torch.Tensor, beam: TimoshenkoBeam)->torch.Tensor:
    """
    Computes the physical loss for the deflection (w) and rotation (psi) fields.

    Args:
        w_model (MLP): An instance of the MLP class representing the neural network for the deflection field.
        psi_model (MLP): An instance of the MLP class representing the neural network for the rotation field.
        X_f_train (torch.Tensor): The input data for the neural networks.
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing the properties of the beam.

    Returns:
        torch.Tensor: The physical loss for the deflection and rotation fields.
    """
    X_f_train.requires_grad = True
    w = w_model(X_f_train)
    psi = psi_model(X_f_train)

    # Compute the derivatives of w and psi
    w_grads = torch.autograd.grad(w, X_f_train, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x = w_grads[:, 0:1]
    w_x_grads = torch.autograd.grad(w_x, X_f_train, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_xx = w_x_grads[:, 0:1]

    psi_grads = torch.autograd.grad(psi, X_f_train, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_x = psi_grads[:, 0:1]
    psi_x_grads = torch.autograd.grad(psi_x, X_f_train, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    psi_xx = psi_x_grads[:, 0:1]

    # Compute the physical loss
    f_w = beam.G*beam.As*(w_xx - psi_x)
    f_psi = beam.E*beam.I*psi_xx + beam.G*beam.As*(w_x - psi)

    return torch.mean(f_w**2) + torch.mean(f_psi**2)

def loss_function(w_model: MLP, psi_model: MLP, bc: TimoshenkoBoundary, bc_pred: TimoshenkoBoundary, X_f_train: torch.Tensor, x_0:torch.Tensor, x_L: torch.Tensor, beam: TimoshenkoBeam)->torch.Tensor:
    """
    Computes the total loss function for the deflection (w) and rotation (psi) fields.

    Args:
        w_model (MLP): An instance of the MLP class representing the neural network for the deflection field.
        psi_model (MLP): An instance of the MLP class representing the neural network for the rotation field.
        bc (TimoshenkoBoundary): An instance of the TimoshenkoBoundary class containing the true boundary conditions.
        bc_pred (TimoshenkoBoundary): An instance of the TimoshenkoBoundary class containing the predicted boundary conditions.
        X_f_train (torch.Tensor): The input data for the neural networks.
        x_0 (torch.Tensor): The input data for the neural networks at the left boundary.
        x_L (torch.Tensor): The input data for the neural networks at the right boundary.
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing the properties of the beam.

    Returns:
        torch.Tensor: The total loss function for the deflection and rotation fields.
    """
    # Predict values at initial boundary conditions
    w_0 = w_model(x_0)
    psi_0 = psi_model(x_0)
    
    # Predict values at final boundary conditions
    w_L = w_model(x_L)
    psi_L = psi_model(x_L)

    # Compute the boundary conditions
    bc_pred = TimoshenkoBoundary(w_0=w_0, w_L=w_L, psi_0=psi_0, psi_L=psi_L)

    # Compute the data loss
    data_loss_val = data_loss(bc, bc_pred)

    # Compute the physical loss
    physical_loss_val = physical_loss(w_model, psi_model, X_f_train, beam)

    return data_loss_val + physical_loss_val

def train_model(w_model: MLP, psi_model: MLP, X_f_train: torch.Tensor, x_0: torch.Tensor, x_L: torch.Tensor, beam: TimoshenkoBeam, bc: TimoshenkoBoundary, epochs: int, lr: float)->Tuple[MLP, MLP]:
    """
    Trains the neural network models for the deflection (w) and rotation (psi) fields of a Timoshenko beam.

    Args:
        w_model (MLP): The neural network model for the deflection field.
        psi_model (MLP): The neural network model for the rotation field.
        X_f_train (torch.Tensor): Training data for the collocation points.
        x_0 (torch.Tensor): Boundary condition at x = 0.
        x_L (torch.Tensor): Boundary condition at x = L.
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing the properties of the beam.
        bc (TimoshenkoBoundary): Boundary conditions for the Timoshenko beam.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple[MLP, MLP]: Trained neural network models for the deflection (w) and rotation (psi) fields.
    """

    # Loss values and epochs
    loss_values = []
    epoch_values = []

    # Define the optimizer
    optimizer = torch.optim.Adam(list(w_model.parameters()) + list(psi_model.parameters()), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Compute the loss function
        loss = loss_function(w_model, psi_model, bc, bc, X_f_train, x_0, x_L, beam)

        # Perform backpropagation
        loss.backward(retain_graph=True)

        # Update the weights
        optimizer.step()

        # Print the loss value
        if epoch % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            loss_values.append(loss.item())
            epoch_values.append(epoch)

    # Plot the loss values
    plot_loss(loss_values, epoch_values)

    return w_model, psi_model

def plot_loss(loss_values: List[float], epoch_values: List[int]):
    """
    Plots the loss values during training.

    Args:
        loss_values (list): A list of loss values.
        epoch_values (list): A list of epoch values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_values, loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_solution(w_model: MLP, psi_model: MLP, beam: TimoshenkoBeam, bc: TimoshenkoBoundary):
    """
    Plots the solution for the deflection (w) and rotation (psi) fields of a Timoshenko beam.

    Args:
        w_model (MLP): The neural network model for the deflection field.
        psi_model (MLP): The neural network model for the rotation field.
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing the properties of the beam.
        bc (TimoshenkoBoundary): Boundary conditions for the Timoshenko beam.
    """
    x_values = np.linspace(0, beam.length, 500)
    w_values = []
    psi_values = []

    for x in x_values:
        w = w_model(torch.tensor([[x]], device=device, dtype=torch.float32)).item()
        psi = psi_model(torch.tensor([[x]], device=device, dtype=torch.float32)).item()
        w_values.append(w)
        psi_values.append(psi)

    # Plot the solution
    plt.figure(figsize=(12, 6))

    # Plot w(x)
    plt.subplot(1, 2, 1)
    plt.plot(x_values, w_values, label='w(x)')
    plt.xlabel('x')
    plt.ylabel('w(x)')
    plt.legend()
    plt.title('Deflection (w)')

    # Plot psi(x)
    plt.subplot(1, 2, 2)
    plt.plot(x_values, psi_values, label='ψ(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('ψ(x)')
    plt.legend()
    plt.title('Rotation (ψ)')

    plt.tight_layout()
    plt.show()

def analyticSolution(x: float, delta: float, beam: TimoshenkoBeam)->Tuple[float, float]:
    """
    Computes the analytical solution for the deflection (w) and rotation (psi) 
    of a Timoshenko beam at a given position x. The following boundary conditions
    are assumed: w(0) = delta, w(L) = 0, psi(0) = 0, psi(L) = 0.

    Args:
        x (float): Position along the beam where the solution is computed.
        delta (float): Scale factor for the deflection and rotation.
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing 
                               the properties of the beam.

    Returns:
        tuple: A tuple containing the deflection (w) and rotation (psi) at position x.
    """
    L = beam.length
    g = 6 * beam.E * beam.I / (beam.G * beam.As * L**2)
    
    # Calculate w(x)
    w = (2 * x**3 / (L**3 * (1 + 2 * g)) - 
         3 * x**2 / (L**2 * (1 + 2 * g)) - 
         2 * x * g / (L * (1 + 2 * g)) + 
         1) * delta
    
    # Calculate psi(x)
    psi = (6 * x**2 / (L**3 * (1 + 2 * g)) - 
           6 * x / (L**2 * (1 + 2 * g))) * delta
    
    return w, psi

def plot_analytical_solution(beam: TimoshenkoBeam, delta: float):
    """
    Plots the analytical solution for the deflection (w) and rotation (psi) of a Timoshenko beam.

    Args:
        beam (TimoshenkoBeam): An instance of the TimoshenkoBeam class containing the properties of the beam.
        delta (float): Scale factor for the deflection and rotation.
    """
    x_values = np.linspace(0, beam.length, 500)
    w_values = []
    psi_values = []

    for x in x_values:
        w, psi = analyticSolution(x, delta, beam)
        w_values.append(w)
        psi_values.append(psi)

    # Plot the analytical solutions
    plt.figure(figsize=(12, 6))

    # Plot w(x)
    plt.subplot(1, 2, 1)
    plt.plot(x_values, w_values, label='w(x)')
    plt.xlabel('x')
    plt.ylabel('w(x)')
    plt.legend()
    plt.title('Deflection (w)')

    # Plot psi(x)
    plt.subplot(1, 2, 2)
    plt.plot(x_values, psi_values, label='ψ(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('ψ(x)')
    plt.legend()
    plt.title('Rotation (ψ)')

    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    beam = TimoshenkoBeam(length=1, G=1, E=1, As=1, I=1)

    # Imposed displacement
    delta = 0.004

    # Hyperparameters
    lr = 0.01
    epochs = 4000
    w_layers =  [1, 20, 20, 20, 20, 1]
    psi_layers =  [1, 20, 20, 20, 20, 1]

    # Boundary conditions
    bc = TimoshenkoBoundary(w_0=delta, w_L=0, psi_0=0, psi_L=0)

    # Number of points
    N_u = 6000
    N_f = 10000 

    # Boundary conditions data
    x_BC_1 = torch.zeros((N_u, 1), device=device, dtype=torch.float32, requires_grad=True)
    x_BC_2 = torch.ones((N_u, 1), device=device, dtype=torch.float32, requires_grad=True) * beam.length

    # Collocation points
    X_f_train = torch.rand((N_f, 1), device=device, dtype=torch.float32) * beam.length
    X_f_train.requires_grad = True

    # Create models
    w_model = w_NN(w_layers)
    psi_model = psi_NN(psi_layers)

    # Train the model
    w_model, psi_model = train_model(w_model=w_model, 
                                    psi_model=psi_model, 
                                    X_f_train=X_f_train, 
                                    x_0=x_BC_1, 
                                    x_L=x_BC_2, 
                                    beam=beam,  
                                    bc=bc, 
                                    epochs=epochs, 
                                    lr=lr)

    # Plot the solution
    plot_solution(w_model, psi_model, beam, bc)

    # Generate data for plotting
    plot_analytical_solution(beam, delta)

if __name__ == "__main__":
    main()



    
