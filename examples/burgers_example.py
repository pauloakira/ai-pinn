# Python libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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

def neural_net(layers):
    return MLP(layers)

def data_loss(u_train, u_pred):
    return torch.mean((u_train - u_pred)**2)

def physical_loss(model, X_f_train, nu):
    X_f_train.requires_grad = True
    u = model(X_f_train)
    grads = torch.autograd.grad(u, X_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 1:2]
    u_t = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x, X_f_train, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)

def loss_function(model, X_u_train, u_train, X_f_train, nu):
    u_pred = model(X_u_train)
    mse_u = data_loss(u_train, u_pred)
    mse_f = physical_loss(model, X_f_train, nu)
    return mse_u + mse_f

def train(model, X_u_train, X_f_train, nu, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, X_u_train, u_train, X_f_train, nu)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == '__main__':
    # Parameters
    nu = 0.01 / np.pi
    layers = [2, 20, 20, 20]

    # Training data
    N_u = 100*2
    N_f = 10000*2

    # Initial boundary conditions: u(0, x) = -sin(pi * x)
    x_IC = np.random.uniform(-1, 1, (N_u, 1))
    t_IC = np.zeros((N_u, 1))
    X_u_train = np.hstack((t_IC, x_IC))
    u_train = -np.sin(np.pi * x_IC)

    # Boundary conditions: u(t, -1) = u(t, 1) = 0
    t_BC = np.random.uniform(0, 1, (N_u, 1))
    x_BC_left = np.ones((N_u, 1)) * -1
    x_BC_right = np.ones((N_u, 1))
    X_u_train = np.vstack((X_u_train, np.hstack((t_BC, x_BC_left)), np.hstack((t_BC, x_BC_right)))) # Combine initial and boundary conditions
    u_train = np.vstack((u_train, np.zeros((N_u, 1)), np.zeros((N_u, 1)))) # Combine initial and boundary conditions

    # Collocation points for PDE residuals
    t_f = np.random.uniform(0, 1, (N_f, 1))
    x_f = np.random.uniform(-1, 1, (N_f, 1))
    X_f_train = np.hstack((t_f, x_f))

    # Convert to torch tensors
    X_u_train = torch.tensor(X_u_train, dtype=torch.float32, requires_grad=True)
    u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True)
    X_f_train = torch.tensor(X_f_train, dtype=torch.float32, requires_grad=True)

    # Create the model
    model = neural_net(layers)
    train(model, X_u_train, X_f_train, nu, 10000, 0.001)

    # Save the model
    torch.save(model.state_dict(), 'models/burgers_continuous_model.pth')

    # Generate test data
    N_test = 1000
    t_test = np.linspace(0, 1, N_test).reshape(-1, 1)
    x_test = np.linspace(-1, 1, N_test).reshape(-1, 1)
    T, X = np.meshgrid(t_test, x_test)
    X_test = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))

    # Convert to torch tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test)
    
    # Reshape
    U_pred = griddata(X_test.detach().numpy(), u_pred.flatten().detach().numpy(), (T, X), method='cubic')
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    for t in np.linspace(0, 1, 5):
        X_test_t = np.hstack((t * np.ones((N_test, 1)), x_test))
        X_test_t = torch.tensor(X_test_t, dtype=torch.float32)
        with torch.no_grad():
            u_pred_t = model(X_test_t)
        plt.plot(x_test, u_pred_t, label=f'PINN Solution at t={t:.2f}')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('PINN Solution of Burgers\' Equation')
    plt.legend()
    plt.grid(True)
    plt.show()
