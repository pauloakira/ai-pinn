import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network model
class BurgersNet(nn.Module):
    def __init__(self, layers):
        super(BurgersNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def forward(self, t, x):
        u = torch.cat([t, x], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            u = torch.tanh(layer(u))
        u = self.layers[-1](u)
        return u

# Define the Burgers' equation
def burgers_equation(t, x, u):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - (0.01 / torch.pi) * u_xx

# Initialize the neural network
layers = [2, 50, 50, 50, 1]
net = BurgersNet(layers)

# Training parameters
learning_rate = 0.0001
num_epochs = 10000
dt = 0.01
q = 4  # Number of Runge-Kutta stages

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Generate initial and boundary training data
t_init = torch.zeros((100, 1), requires_grad=True)
x_init = torch.linspace(-1, 1, 100).view(-1, 1)
u_init = -torch.sin(torch.pi * x_init)

x_bound = torch.tensor([[-1.0], [1.0]], requires_grad=True)
t_bound = torch.linspace(0, 1, 100).view(-1, 1)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Generate collocation points for the physical loss
    t_colloc = torch.rand((100, 1), requires_grad=True)
    x_colloc = torch.rand((100, 1), requires_grad=True)
    u_colloc = net(t_colloc, x_colloc)
    
    # Compute the physical loss
    phys_loss = torch.mean((burgers_equation(t_colloc, x_colloc, u_colloc))**2)
    
    # Compute the data loss for initial condition
    u_init_pred = net(t_init, x_init)
    init_loss = torch.mean((u_init_pred - u_init)**2)
    
    # Compute the boundary loss for each Runge-Kutta stage
    bound_loss = 0
    for i in range(1, q + 1):
        t_rk_stage = t_bound + i * dt / q
        u_bound_pred_left = net(t_rk_stage, x_bound[0].repeat(100, 1))
        u_bound_pred_right = net(t_rk_stage, x_bound[1].repeat(100, 1))
        bound_loss += torch.mean(u_bound_pred_left**2) + torch.mean(u_bound_pred_right**2)
    
    # Add the boundary loss for the final step
    u_bound_pred_left = net(t_bound + dt, x_bound[0].repeat(100, 1))
    u_bound_pred_right = net(t_bound + dt, x_bound[1].repeat(100, 1))
    bound_loss += torch.mean(u_bound_pred_left**2) + torch.mean(u_bound_pred_right**2)
    
    # Total loss
    loss = phys_loss + init_loss + bound_loss
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {loss.item():.4f}, Phys Loss: {phys_loss.item():.4f}, Init Loss: {init_loss.item():.4f}, Bound Loss: {bound_loss.item():.4f}')

# Generate predictions over a grid of (t, x) values
t_grid = torch.linspace(0, 1, 100).view(-1, 1)
x_grid = torch.linspace(-1, 1, 100).view(-1, 1)
T, X = torch.meshgrid(t_grid.squeeze(), x_grid.squeeze())
T = T.reshape(-1, 1)
X = X.reshape(-1, 1)
U_pred = net(T, X).detach().numpy()
U_pred = U_pred.reshape(100, 100)

# Plot the predicted solution
plt.figure(figsize=(10, 6))
times = [0.0, 0.1, 0.2, 0.5, 1.0]
for t in times:
    X_test_t = torch.tensor(np.hstack((t * np.ones((100, 1)), x_grid)), dtype=torch.float32)
    with torch.no_grad():
        u_pred_t = net(t * torch.ones(100, 1), x_grid).detach().numpy()
    plt.plot(x_grid.numpy(), u_pred_t, label=f't={t:.2f}')

plt.xlabel('x', fontsize=14)
plt.ylabel('u', fontsize=14)
plt.title('PINN Solution of Burgers\' Equation', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
