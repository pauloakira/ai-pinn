import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from core.parabolic.vem import VEMParabolic

# Create a simple square mesh with 4 triangular elements
vertices = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

elements = np.array([
    [0, 1, 2],  # Triangle 1
    [0, 2, 3]   # Triangle 2
])

mesh = {
    'vertices': vertices,
    'elements': elements
}

# Initialize solver
vem = VEMParabolic(mesh, k=1)

# Define initial condition
u0 = np.zeros(len(vertices))
for i, (x, y) in enumerate(vertices):
    u0[i] = np.sin(np.pi*x) * np.sin(np.pi*y)

# Define source term
def f(t, x, y):
    return np.exp(t) * np.sin(np.pi*x) * np.sin(np.pi*y)

# Solve
T = 1.0  # Final time
nt = 100  # Number of time steps
U = vem.solve_backward_euler(T, nt, u0, f)

print(f"Solution shape: {U.shape}")  # (nt+1) x n_vertices

print(U)