# Python libs
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Custom libs
from utils.mlflow_helper import mlflowPipeline

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Setting up Metal
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cpu')
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
    psi_x_train: torch.Tensor
    psi_xx_train: torch.Tensor

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
        z = self.linears[-1](z)
        return z

# class MLP(nn.Module):
#     def __init__(self, layers):
#         super(MLP, self).__init__()
#         self.layers = layers
#         self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
#         self.dropout = nn.Dropout(0.1)
#         self.output = nn.Linear(layers[-1], 1)

#     def forward(self, x):
#         z = x
#         for i in range(len(self.linears)-1):
#             z = torch.relu(self.linears[i](z))
#             z = self.dropout(z)
#         z = self.linears[-1](z)
#         return z
    
def w_NN(layers):
    return MLP(layers).to(device)

def psi_NN(layers):
    return MLP(layers).to(device)

def data_loss(bc: BoundaryConditions, w_pred, psi_pred, psi_x_pred, psi_xx_pred):
    return (torch.mean((bc.w_train - w_pred)**2) +
            torch.mean((bc.psi_train - psi_pred)**2) +
            torch.mean((bc.psi_x_train - psi_x_pred)**2) +
            torch.mean((bc.psi_xx_train - psi_xx_pred)**2))

def physical_loss(w_model, psi_model, X_f_train, G, E, As, I, q):
    X_f_train.requires_grad = True
    # Models
    w = w_model(X_f_train)
    psi = psi_model(X_f_train)
    # w derivatives
    w_grads = torch.autograd.grad(w, X_f_train, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x = w_grads[:, 0:1]
    # psi derivatives
    psi_grads = torch.autograd.grad(psi, X_f_train, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_x = psi_grads[:, 0:1]
    psi_x_grads = torch.autograd.grad(psi_x, X_f_train, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    psi_xx = psi_x_grads[:, 0:1]
    psi_xx_grads = torch.autograd.grad(psi_xx, X_f_train, grad_outputs=torch.ones_like(psi_xx), create_graph=True)[0]
    psi_xxx = psi_xx_grads[:, 0:1]
    # Physical-informed equations
    f_w = w_x - psi_x + 1/(G*As)*E*I*psi_xx
    f_psi = E*I*psi_xxx - q(X_f_train)
    return torch.mean(f_w**2) + torch.mean(f_psi**2)

def loss_function(w_model, psi_model, X_f_train, G, E, As, I, q, bc: BoundaryConditions):
    # Predict values at boundary conditions
    w_pred = w_model(bc.x_BC_1)
    psi_pred = psi_model(bc.x_BC_1)
    psi_x_pred = torch.autograd.grad(psi_pred, bc.x_BC_1, grad_outputs=torch.ones_like(psi_pred), create_graph=True)[0]
    psi_xx_pred = torch.autograd.grad(psi_x_pred, bc.x_BC_1, grad_outputs=torch.ones_like(psi_x_pred), create_graph=True)[0]
    
    # Calculate boundary condition loss
    mse_bc = data_loss(bc, w_pred, psi_pred, psi_x_pred, psi_xx_pred)
    
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
        loss.backward()
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
    F = -4

    # Hyperparameters
    lr = 0.01
    epochs = 10000
    # w_layers =  [1, 100, 100, 50, 20, 1]
    # psi_layers =  [1, 100, 100, 50, 20, 1]
    w_layers =  [1, 256, 128, 128, 64, 64, 32, 16, 8, 1]
    psi_layers =  [1, 200, 200, 50, 20, 1]

    # Training data
    N_u = 25000*1
    N_f = 60000*1

    # Boundary conditions: w(0) = 0, psi(0) = 0
    x_BC_1 = torch.zeros((N_u, 1), device=device, dtype=torch.float32, requires_grad=True)
    w_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)
    psi_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)

    # Boundary condtions: psi_x(L) = 0, psi_xx(L) = F/(EI)
    x_BC_2 = torch.ones((N_u, 1), device=device, dtype=torch.float32) * length
    psi_x_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)
    psi_xx_train = torch.ones((N_u, 1), device=device, dtype=torch.float32) * (F / (E * I))

    # Collocation points
    X_f_train = torch.rand((N_f, 1), device=device, dtype=torch.float32) * length

    # Boundary conditions dataclass
    bc = BoundaryConditions(x_BC_1=x_BC_1, w_train=w_train, psi_train=psi_train,
                            x_BC_2=x_BC_2, psi_x_train=psi_x_train, psi_xx_train=psi_xx_train)

    # Create the model
    w_model = w_NN(w_layers)
    psi_model = psi_NN(psi_layers)
    
    # Train the model
    start_time = time.time()
    train(w_model, psi_model, X_f_train, G, E, As, I, lambda x: torch.zeros_like(x), bc, epochs, lr, step_lr)
    execution_time = time.time() - start_time

    # Save the model
    #torch.save(w_model.state_dict(), f'models/timoshenko_w_model_{timestamp}.pth')
    # torch.save(psi_model.state_dict(), f'models/timoshenko_psi_model_{timestamp}.pth')

    # Generate predictions
    x_test = torch.linspace(0, length, 100, device=device).view(-1, 1)
    # w_pred = w_model(x_test).detach().numpy()
    w_pred_tensor = w_model(x_test).detach()
    w_pred = w_pred_tensor.cpu().numpy()
    # psi_pred = psi_model(x_test).detach().numpy()
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
