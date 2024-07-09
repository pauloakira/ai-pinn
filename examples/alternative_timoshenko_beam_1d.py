# import time
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from dataclasses import dataclass
# from datetime import datetime

# # Custom libs
# from utils.mlflow_helper import mlflowPipeline

# # Set the random seed for reproducibility
# np.random.seed(42)
# torch.manual_seed(42)

# # Setting up Metal
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Defining model objects
# @dataclass
# class TimoshenkoBeam:
#     length: float
#     E: float
#     I: float
#     kappa: float
#     A: float
#     G: float

# @dataclass
# class BoundaryConditions:
#     x_BC_1: torch.Tensor
#     w_train: torch.Tensor
#     x_BC_2: torch.Tensor
#     w_x_train: torch.Tensor
#     w_xxx_train: torch.Tensor

# class MLP(nn.Module):
#     def __init__(self, layers):
#         super(MLP, self).__init__()
#         self.layers = layers
#         self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
#         self.output = nn.Linear(layers[-1], 1)

#     def forward(self, x):
#         z = x
#         for i in range(len(self.linears)-1):
#             z = torch.tanh(self.linears[i](z))
#         z = self.linears[-1](z)
#         return z

# def w_NN(layers):
#     return MLP(layers).to(device)

# def data_loss(bc: BoundaryConditions, w_pred, w_x_pred, w_xxx_pred):
#     return (torch.mean((bc.w_train - w_pred)**2) +
#             torch.mean((bc.w_x_train - w_x_pred)**2) +
#             torch.mean((bc.w_xxx_train - w_xxx_pred)**2))

# def physical_loss(w_model, X_f_train, E, I, kappa, A, G):
#     X_f_train.requires_grad = True
#     # Model
#     w = w_model(X_f_train)
#     # w derivatives
#     w_grads = torch.autograd.grad(w, X_f_train, grad_outputs=torch.ones_like(w), create_graph=True)[0]
#     w_x = w_grads[:, 0:1]
#     w_x_grads = torch.autograd.grad(w_x, X_f_train, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
#     w_xx = w_x_grads[:, 0:1]
#     w_xx_grads = torch.autograd.grad(w_xx, X_f_train, grad_outputs=torch.ones_like(w_xx), create_graph=True)[0]
#     w_xxx = w_xx_grads[:, 0:1]
#     w_xxx_grads = torch.autograd.grad(w_xxx, X_f_train, grad_outputs=torch.ones_like(w_xxx), create_graph=True)[0]
#     w_xxxx = w_xxx_grads[:, 0:1]
    
#     # Physical-informed equation
#     f_w = E * I * w_xxxx
#     return torch.mean(f_w**2)

# def loss_function(w_model, X_f_train, E, I, kappa, A, G, bc: BoundaryConditions):
#     # Predict values at initial boundary conditions
#     w_pred = w_model(bc.x_BC_1)
#     w_x_pred = torch.autograd.grad(w_pred, bc.x_BC_1, grad_outputs=torch.ones_like(w_pred), create_graph=True)[0]

#     # Predict values at end boundary conditions
#     w_end_pred = w_model(bc.x_BC_2)
#     w_x_end_pred = torch.autograd.grad(w_end_pred, bc.x_BC_2, grad_outputs=torch.ones_like(w_end_pred), create_graph=True)[0]
#     w_xx_end_pred = torch.autograd.grad(w_x_end_pred, bc.x_BC_2, grad_outputs=torch.ones_like(w_x_end_pred), create_graph=True)[0]
#     w_xxx_end_pred = torch.autograd.grad(w_xx_end_pred, bc.x_BC_2, grad_outputs=torch.ones_like(w_xx_end_pred), create_graph=True)[0]

#     # Calculate boundary condition loss
#     mse_bc = data_loss(bc, w_pred, w_x_end_pred, w_xxx_end_pred)
    
#     # Calculate physical loss
#     mse_f = physical_loss(w_model, X_f_train, E, I, kappa, A, G)
    
#     return mse_bc + mse_f

# def train(w_model, X_f_train, E, I, kappa, A, G, bc: BoundaryConditions, epochs, learning_rate, step_lr):
#     optimizer_w = torch.optim.Adam(w_model.parameters(), lr=learning_rate)
#     previous_loss = 0

#     if step_lr:
#         scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=1000, gamma=0.1)
        
#     for epoch in range(epochs):
#         optimizer_w.zero_grad()
#         loss = loss_function(w_model, X_f_train, E, I, kappa, A, G, bc)
#         loss.backward(retain_graph=True)
#         optimizer_w.step()

#         # Step the scheduler
#         if step_lr:
#             scheduler_w.step()
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item()}")
#             if previous_loss == loss.item():
#                 print(f"Converged at epoch {epoch}, Loss: {loss.item()}")
#                 break
#             previous_loss = loss.item()

# if __name__ == '__main__':
#     # Datetime for logging
#     timestamp = int(datetime.now().timestamp())

#     # Setup
#     step_lr = False

#     # Parameters
#     length = 100
#     E = 1e7
#     I = 1/12
#     kappa = 5/6  # Example value
#     A = 1
#     G = 5e6
#     F = -0.4

#     # Hyperparameters
#     lr = 0.0001
#     epochs = 2000
#     w_layers =  [1, 256, 128, 128, 64, 64, 32, 16, 8, 1]

#     # Training data
#     mult_constant = 4
#     N_u = 2500 * mult_constant
#     N_f = 10000 * mult_constant

#     # Boundary conditions: w(0) = 0
#     x_BC_1 = torch.zeros((N_u, 1), device=device, dtype=torch.float32, requires_grad=True)
#     w_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)

#     # Boundary conditions: w_x(L) = 0, w_xxx(L) = F
#     x_BC_2 = torch.ones((N_u, 1), device=device, dtype=torch.float32, requires_grad=True) * length
#     w_x_train = torch.zeros((N_u, 1), device=device, dtype=torch.float32)
#     w_xxx_train = torch.ones((N_u, 1), device=device, dtype=torch.float32) * F

#     # Collocation points
#     X_f_train = torch.rand((N_f, 1), device=device, dtype=torch.float32) * length
#     X_f_train.requires_grad = True

#     # Boundary conditions dataclass
#     bc = BoundaryConditions(x_BC_1=x_BC_1, w_train=w_train,
#                             x_BC_2=x_BC_2, w_x_train=w_x_train, w_xxx_train=w_xxx_train)

#     # Create the model
#     w_model = w_NN(w_layers)
    
#     # Train the model
#     start_time = time.time()
#     train(w_model, X_f_train, E, I, kappa, A, G, bc, epochs, lr, step_lr)
#     execution_time = time.time() - start_time

#     # Save the model
#     # torch.save(w_model.state_dict(), f'models/timoshenko_w_model_{timestamp}.pth')

#     # Generate predictions
#     x_test = torch.linspace(0, length, 100, device=device).view(-1, 1)
#     w_pred_tensor = w_model(x_test).detach()
#     w_pred = w_pred_tensor.cpu().numpy()

#     print(f"Analytical at x=L: w(L) = {F*length**3/(3*E*I)}")
#     print(f"Predicted at x=L: w(L) = {w_pred[-1][0]}")

#     log_object = {
#         "length": length,
#         "E": E,
#         "I": I,
#         "kappa": kappa,
#         "A": A,
#         "G": G,
#         "F": F,
#         "lr": lr,
#         "epochs": epochs,
#         "N_u": N_u,
#         "N_f": N_f,
#         "w_pred": w_pred[-1][0],
#         "w_analytical": F*length**3/(3*E*I),
#         "w_error": abs(w_pred[-1][0] - F*length**3/(3*E*I))/abs(F*length**3/(3*E*I)),
#         "training_time": execution_time,
#         "w_layers": w_layers,
#         "step_lt": step_lr
#     }

#     # Log the results
#     mlflowPipeline("TimoshenkoBeam", f"TimoshenkoBeam_{timestamp}", log_object)

#     # Converting back to cpu
#     x_test = x_test.cpu()

#     # Plot the results
#     plt.figure(figsize=(10, 5))
#     plt.plot(x_test.numpy(), w_pred, label='w(x)')
#     plt.xlabel('x')
#     plt.ylabel('Deflection w(x)')
#     plt.legend()
#     plt.title('Solution of the Differential Equation')
    
#     # Save the figure
#     plt.savefig(f'images/alt_timoshenko_beam_{timestamp}.png')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Setting up Metal
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

def weak_form_loss(w_nn, psi_nn, x, q, G, As, E, I):
    w = w_nn(x)
    psi = psi_nn(x)
    
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    
    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    
    residual_w = w_xx - psi - q(x) / (G * As)
    residual_psi = psi_xx + psi / (E * I)
    
    test_funcs_w = [x * (1 - x), torch.sin(torch.pi * x)]
    test_funcs_psi = [x * (1 - x), torch.sin(torch.pi * x)]
    
    loss = 0
    for vw in test_funcs_w:
        integrand_w = residual_w * vw
        loss += torch.mean(integrand_w**2)
        
    for vpsi in test_funcs_psi:
        integrand_psi = residual_psi * vpsi
        loss += torch.mean(integrand_psi**2)
    
    return loss

def train_model(w_model, psi_model, x_train, q, G, As, E, I, epochs, lr):
    optimizer_w = optim.Adam(w_model.parameters(), lr=lr)
    optimizer_psi = optim.Adam(psi_model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer_w.zero_grad()
        optimizer_psi.zero_grad()
        loss = weak_form_loss(w_model, psi_model, x_train, q, G, As, E, I)
        loss.backward()
        optimizer_w.step()
        optimizer_psi.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Example Usage
x_train = torch.linspace(0, 1, 100).view(-1, 1).to(device).requires_grad_(True)
q = lambda x: torch.sin(torch.pi * x)  # Example distributed load

# Neural Network configuration
layers = [1, 50, 50, 1]
w_model = NeuralNetwork(layers).to(device)
psi_model = NeuralNetwork(layers).to(device)

# Parameters
G = 5e6
As = 1
E = 1e7
I = 1/12

# Train the model
train_model(w_model, psi_model, x_train, q, G, As, E, I, epochs=1000, lr=0.001)

# Generate Predictions
def generate_predictions(w_model, psi_model, x_test):
    w_pred = w_model(x_test).detach().cpu().numpy()
    psi_pred = psi_model(x_test).detach().cpu().numpy()
    return w_pred, psi_pred

# Prepare test points
x_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)

# Generate predictions
w_pred, psi_pred = generate_predictions(w_model, psi_model, x_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_test.cpu().numpy(), w_pred, label='Predicted $w(x)$')
plt.xlabel('$x$')
plt.ylabel('$w(x)$')
plt.legend()
plt.title('Transverse Displacement')

plt.subplot(1, 2, 2)
plt.plot(x_test.cpu().numpy(), psi_pred, label='Predicted $\\psi(x)$')
plt.xlabel('$x$')
plt.ylabel('$\\psi(x)$')
plt.legend()
plt.title('Rotation')

plt.tight_layout()
plt.show()





