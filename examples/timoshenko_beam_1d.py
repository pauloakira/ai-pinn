# Python libs
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from dataclasses import dataclass
from datetime import datetime

# Custom libs
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
    x_BC_1: torch.Tensor
    w_train: torch.Tensor
    psi_train: torch.Tensor
    x_BC_2: torch.Tensor
    M_train: torch.Tensor
    V_train: torch.Tensor

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

def w_NN(layers)->MLP:
    return MLP(layers).to(device)

def psi_NN(layers)->MLP:
    return MLP(layers).to(device)

def data_loss(bc: BoundaryConditions, w_pred, psi_pred, M_pred, V_pred)->tuple:
    return (torch.mean((bc.w_train - w_pred)**2) +
            torch.mean((bc.psi_train - psi_pred)**2) +
            torch.mean((bc.M_train - M_pred)**2) +
            torch.mean((bc.V_train - V_pred)**2))

def physical_loss(w_model, psi_model, X_f_train, G, E, As, I, q)->torch.Tensor:
    X_f_train.requires_grad = True
    # Models
    w = w_model(X_f_train)
    psi = psi_model(X_f_train)
    # w derivatives
    w_grads = torch.autograd.grad(w, X_f_train, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x = w_grads[:, 0:1]
    w_x_grads = torch.autograd.grad(w_x, X_f_train, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_xx = w_x_grads[:, 0:1]
    # psi derivatives
    psi_grads = torch.autograd.grad(psi, X_f_train, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_x = psi_grads[:, 0:1]
    psi_x_grads = torch.autograd.grad(psi_x, X_f_train, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    psi_xx = psi_x_grads[:, 0:1]
    # Physical-informed equations
    f_w = G*As*(w_xx - psi_x) + q(X_f_train)
    f_psi = E*I*psi_xx + G*As*(w_x - psi)
    return torch.mean(f_w**2) + torch.mean(f_psi**2)

def loss_function(w_model, psi_model, X_f_train, G, E, As, I, q, bc: BoundaryConditions)->torch.Tensor:
    # Predict values at initial boundary conditions
    w_pred = w_model(bc.x_BC_1)
    psi_pred = psi_model(bc.x_BC_1)
    
    # Predict values at end boundary conditions
    psi_end_pred = psi_model(bc.x_BC_2)
    M_end_pred = E * I * psi_end_pred
    V_end_pred = torch.autograd.grad(M_end_pred, bc.x_BC_2, grad_outputs=torch.ones_like(M_end_pred), create_graph=True)[0]
    
    # Calculate boundary condition loss
    mse_bc = data_loss(bc, w_pred, psi_pred, M_end_pred, V_end_pred)
    
    # Calculate physical loss
    mse_f = physical_loss(w_model, psi_model, X_f_train, G, E, As, I, q)
    
    return mse_bc + mse_f

def train(w_model, psi_model, X_f_train, G, E, As, I, q, bc: BoundaryConditions, epochs, learning_rate, step_lr):
    optimizer_w = torch.optim.Adam(w_model.parameters(), lr=learning_rate)
    optimizer_psi = torch.optim.Adam(psi_model.parameters(), lr=learning_rate)

    if step_lr:
        scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=1000, gamma=0.1)
        scheduler_psi = torch.optim.lr_scheduler.StepLR(optimizer_psi, step_size=1000, gamma=0.1)
        
    for epoch in range(epochs):
        optimizer_w.zero_grad()
        optimizer_psi.zero_grad()
        loss = loss_function(w_model, psi_model, X_f_train, G, E, As, I, q, bc)
        loss.backward(retain_graph=True)
        optimizer_w.step()
        optimizer_psi.step()

        # Step the scheduler
        if step_lr:
            scheduler_w.step()
            scheduler_psi.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':
    # Datetime for logging
    timestamp = int(datetime.now().timestamp())

    # Setup
    step_lr = False

    # Parameters
    length = 100
    G = 5e6
    E = 1e7
    As = 1
    I = 1/12
    F = 0.04

    # Hyperparameters
    lr = 0.01
    epochs = 8000
    w_layers =  [1, 50, 50, 20, 1]
    psi_layers =  [1, 50, 50, 20, 1]

    # Training data
    mult_constant = 3
    N_u = 4000 * mult_constant
    N_f = 10000 * mult_constant

    # Boundary conditions: w(0) = 0, psi(0) = 0
    x_BC_1 = torch.zeros((N_u, 1), device=device, dtype=torch.float32, requires_grad=True)
    w_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)
    psi_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)

    # Boundary conditions: M(L) = 0, V(L) = F
    x_BC_2 = torch.ones((N_u, 1), device=device, dtype=torch.float32, requires_grad=True) * length
    M_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)
    V_train = torch.ones((N_u, 1), device=device, dtype=torch.float32) * F

    # Collocation points
    X_f_train = torch.rand((N_f, 1), device=device, dtype=torch.float32) * length
    X_f_train.requires_grad = True

    # Boundary conditions dataclass
    bc = BoundaryConditions(x_BC_1=x_BC_1, w_train=w_train, psi_train=psi_train,
                            x_BC_2=x_BC_2, M_train=M_train, V_train=V_train)

    # Create the model
    w_model = w_NN(w_layers)
    psi_model = psi_NN(psi_layers)
    
    # Train the model
    start_time = time.time()
    train(w_model, psi_model, X_f_train, G, E, As, I, lambda x: torch.zeros_like(x), bc, epochs, lr, step_lr)
    execution_time = time.time() - start_time

    # Save the model
    # torch.save(w_model.state_dict(), f'models/timoshenko_w_model_{timestamp}.pth')
    # torch.save(psi_model.state_dict(), f'models/timoshenko_psi_model_{timestamp}.pth')

    # Generate predictions
    x_test = torch.linspace(0, length, 100, device=device).view(-1, 1)
    w_pred_tensor = w_model(x_test).detach()
    w_pred = w_pred_tensor.cpu().numpy()
    psi_pred_tensor = psi_model(x_test).detach()
    psi_pred = psi_pred_tensor.cpu().numpy()

    print(f"Analytical at x=L: w(L) = {F*length**3/(3*E*I)}, psi(L) = {F*length**2/(3*E*I)}")
    print(f"Predicted at x=L: w(L) = {w_pred[-1][0]}, psi(L) = {psi_pred[-1][0]}")

    log_object = {
        "length": length,
        "G": G,
        "E": E,
        "As": As,
        "I": I,
        "F": F,
        "lr": lr,
        "epochs": epochs,
        "N_u": N_u,
        "N_f": N_f,
        "w_pred": w_pred[-1][0],
        "psi_pred": psi_pred[-1][0],
        "w_analytical": F*length**3/(3*E*I),
        "psi_analytical": F*length**2/(3*E*I),
        "w_error": abs(w_pred[-1][0] - F*length**3/(3*E*I))/abs(F*length**3/(3*E*I)),
        "psi_error": abs(psi_pred[-1][0] - F*length**2/(3*E*I))/abs(F*length**2/(3*E*I)),
        "training_time": execution_time,
        "w_layers": w_layers,
        "psi_layers": psi_layers,
        "step_lt": step_lr
    }

    # Log the results
    mlflowPipeline("TimoshenkoBeam", f"TimoshenkoBeam_{timestamp}", log_object)

    # Converting back to cpu
    x_test = x_test.cpu()

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(x_test.numpy(), w_pred, label='w(x)')
    plt.plot(x_test.numpy(), psi_pred, label='ψ(x)')
    plt.xlabel('x')
    plt.ylabel('Function values')
    plt.legend()
    plt.title('Solutions of the differential equations')
    
    # Save the figure
    plt.savefig(f'images/timoshenko_beam_{timestamp}.png')

