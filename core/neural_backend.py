import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import core.vem as vem
from typing import Tuple, List

import core.loss as loss_function
import core.errors as errors

# Define neural network for the beam problem
class BeamApproximator(nn.Module):
    def __init__(self, input_dim, layers, ndof):
        super(BeamApproximator, self).__init__()
        # First layer from input to the first hidden layer
        self.fin = nn.Linear(input_dim, layers[0])
        
        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # Final output layer from the last hidden layer to the output (ndof)
        self.fout = nn.Linear(layers[-1], ndof)

    def forward(self, x):
        # Pass through the first layer
        z = torch.relu(self.fin(x))
        
        # Pass through the hidden layers
        for layer in self.hidden:
            z = torch.sigmoid(layer(z))
            # z = torch.nn.functional.leaky_relu(layer(z), negative_slope=0.01)
        
        # Final output layer
        z = self.fout(z)
        
        return z
    

class ResidualBeamApproximator(nn.Module):
    def __init__(self, input_dim, layers, ndof):
        super(ResidualBeamApproximator, self).__init__()
        # First layer from input to the first hidden layer
        self.fin = nn.Linear(input_dim, layers[0])
        
        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # Final output layer from the last hidden layer to the output (ndof)
        self.fout = nn.Linear(layers[-1], ndof)

    def forward(self, x):
        # Pass through the first layer
        z = torch.relu(self.fin(x))
        
        # Pass through the hidden layers
        for layer in self.hidden:
            z_res = z
            z = torch.sigmoid(layer(z))

            # Residual block
            if z.shape == z_res.shape:
                z = z + z_res
        
        # Final output layer
        z = self.fout(z)
        
        return z


class BeamApproximatorWithMaterials(nn.Module):
    def __init__(self, input_dim_nodes, input_dim_materials, nodes_layers, material_layers, final_layers, ndof):
        super(BeamApproximatorWithMaterials, self).__init__()
        # Neural network for nodes
        self.nodes_in = nn.Linear(input_dim_nodes, nodes_layers[0])
        self.nodes_hidden = nn.ModuleList([nn.Linear(nodes_layers[i], nodes_layers[i+1]) for i in range(len(nodes_layers)-1)])
        self.nodes_out = nn.Linear(nodes_layers[-1], ndof)

        # Neural network for materials
        self.materials_in = nn.Linear(input_dim_materials, material_layers[0])
        self.materials_hidden = nn.ModuleList([nn.Linear(material_layers[i], material_layers[i+1]) for i in range(len(material_layers)-1)])
        self.materials_out = nn.Linear(material_layers[-1], ndof)

        # Final output layer
        self.final_in = nn.Linear(ndof*2, final_layers[0])
        self.final_hidden = nn.ModuleList([nn.Linear(final_layers[i], final_layers[i+1]) for i in range(len(final_layers)-1)])
        self.final_out = nn.Linear(final_layers[-1], ndof)  # Output layer

    def forward(self, x_nodes, x_materials):
        # Pass through the first layer
        z_nodes = torch.relu(self.nodes_in(x_nodes))
        z_materials = torch.relu(self.materials_in(x_materials))

        # Pass through the hidden layers
        for layer in self.nodes_hidden:
            z_nodes = torch.relu(layer(z_nodes))
        z_nodes = self.nodes_out(z_nodes)

        for layer in self.materials_hidden:
            z_materials = torch.relu(layer(z_materials))
        z_materials = self.materials_out(z_materials)

        # Concatenate the nodes and materials
        z_combined = torch.cat([z_nodes, z_materials])

        # Pass through the final layers
        z_combined = torch.relu(self.final_in(z_combined))
        for layer in self.final_hidden:
            z_combined = torch.relu(layer(z_combined))
        z_combined = self.final_out(z_combined)
        
        return z_combined


def normalize_inputs(nodes: torch.Tensor, material_params: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    if(isinstance(nodes, np.ndarray)):
        nodes = torch.tensor(nodes, dtype=torch.float32)
    if(isinstance(material_params, np.ndarray)):
        material_params = torch.tensor(material_params, dtype=torch.float32)
    nodes = nodes.flatten()
    material_params = material_params.flatten()
    mu_nodes = torch.mean(nodes)
    sigma_nodes = torch.std(nodes)
    mu_material_params = torch.mean(material_params)
    sigma_material_params = torch.std(material_params)

    normalized_nodes = (nodes - mu_nodes) / sigma_nodes
    normalized_material_params = (material_params - mu_material_params) / sigma_material_params

    # Convert to tensor
    normalized_nodes = torch.tensor(normalized_nodes, dtype=torch.float32, requires_grad=True)
    normalized_material_params = torch.tensor(normalized_material_params, dtype=torch.float32, requires_grad=True)

    return normalized_nodes, normalized_material_params


def train_material_portic(epochs: int, 
                   nodes, 
                   K: np.array, 
                   f: np.array, 
                   E: float, 
                   A: float, 
                   I: float, 
                   uh_vem: np.array, 
                   nodes_layers: List[int],
                   material_layers: List[int],
                   final_layers: List[int],
                   verbose=True, 
                   noramlize_inputs=False, 
                   network_type='material'):
    ndof = 3 * len(nodes)
    input_dim = 2*len(nodes) + 3

    input_dim_nodes = 2*len(nodes)
    input_dim_materials = 3

    # Original material parameters
    material_params_1 = torch.tensor([E, A, I], dtype=torch.float32)
    print(f"Material params shape: {material_params_1.shape}")

    # Perturbed material parameters (slightly changed)
    material_params_2 = torch.tensor([E *1.1, A * 1.1, I * 0.9], dtype=torch.float32)

    if noramlize_inputs:
        nodes, material_params_1 = normalize_inputs(nodes, material_params_1)
        _, material_params_2 = normalize_inputs(nodes, material_params_2)

    nodes = nodes.flatten()
    nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)
    print(f"Nodes shape: {nodes.shape}")

    input_vector = torch.cat([nodes, material_params_1])

    lr = 1e-3

    # Initialize the model and optimizer
    if network_type == 'residual':
        # layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024]
        # layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048]
        # layers = [128, 128, 256, 256, 512, 512, 512, 512]
        # layers = [128, 256, 512]
        concatanate=True
        model = ResidualBeamApproximator(input_dim, nodes_layers, ndof)
    if network_type == 'material':
        # nodes_layers = [128, 256, 512, 512, 512, 512]  # Layers for nodes sub-network
        # material_layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048] # Layers for materials sub-network
        # final_layers = [1024, 1024, 1024, 1024]  # Layers for final combination network
        # Concatanete the nodes and materials
        concatanate = False
        model = BeamApproximatorWithMaterials(
            input_dim_nodes=input_dim_nodes, 
            input_dim_materials=input_dim_materials, 
            nodes_layers=nodes_layers, 
            material_layers=material_layers, 
            final_layers=final_layers, 
            ndof=ndof)
    else:
        layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024]
        # layers = [128, 128, 256, 256, 512, 512, 512, 512]
        # layers = [128, 256, 512]
        concatanate=True
        model = BeamApproximator(input_dim, layers, ndof)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.0000000001, weight_decay=1e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.0000000001)

    K = torch.tensor(K, dtype=torch.float32, requires_grad=True)
    f = torch.tensor(f, dtype=torch.float32, requires_grad=True)

    total_loss_values = []
    loss_values = []
    material_loss_values = []
    sobolev_loss_values = []
    alpha_values_values = []

    # Scaling factor for loss
    # alpha = 1e-17

    for epoch in range(epochs):
        optimizer.zero_grad()
        # uh = model(input_vector)
        uh = model(nodes, material_params_1)
        
        # Compute the loss
        loss = loss_function.compute_loss_with_uh(uh_vem, uh)
        # Compute the sobolev loss
        sobolev_loss = loss_function.compute_sobolev_loss(model, nodes, material_params_1,loss, concatanate)
        # Compute material penalty
        material_penalty = loss_function.compute_material_penalty(model, nodes, material_params_1, material_params_2, concatanate)
        # Normalize the loss and penalty
        alpha = loss_function.normalize_loss_and_penalty(loss, material_penalty)
        total_loss = loss + alpha * material_penalty + sobolev_loss
        
        total_loss.backward()
        
        optimizer.step()
        if epoch > 0:
            total_loss_values.append(total_loss.item())
            loss_values.append(loss.item())
            material_loss_values.append(material_penalty.item())
            sobolev_loss_values.append(sobolev_loss.item())
            alpha_values_values.append(alpha)
        
        if verbose:
            print(f'Epoch: {epoch + 1}, Total Loss: {total_loss.item()}')
    
    if verbose:
        print("Training complete.")
        plt.plot(total_loss_values)
        plt.xlabel('Epochs (Sub-Epochs)')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.show()

    return input_vector, model, total_loss_values, loss_values, material_loss_values, sobolev_loss_values, alpha_values_values


def test_portic(nodes, material_params, model, uh_vem, K, f, concatanate=False):
    # Set the model to evaluation mode
    model.eval()

    # Ensure nodes and material_params are torch tensors
    if isinstance(nodes, np.ndarray):
        nodes = torch.tensor(nodes, dtype=torch.float32)
    if isinstance(material_params, np.ndarray):
        material_params = torch.tensor(material_params, dtype=torch.float32)

    K = torch.tensor(K, dtype=torch.float32, requires_grad=True)
    f = torch.tensor(f, dtype=torch.float32, requires_grad=True)
    uh_vem = torch.tensor(uh_vem, dtype=torch.float32)

    # Ensure gradients are not tracked during prediction
    with torch.no_grad():
        # Use the trained model to make predictions
        if concatanate:
            input_vector = torch.cat((nodes, material_params))
            predicted_displacements = model(input_vector)
        else:
            predicted_displacements = model(nodes, material_params)

    # Print or use the predicted displacements
    print("Predicted displacements:", predicted_displacements)

    # Compute errors and ensure tensors are on the same device
    l2_error = errors.compute_l2_error(uh_vem, predicted_displacements).item()
    energy_error = errors.compute_energy_error(K, uh_vem, predicted_displacements).item()
    h1_error = errors.compute_h1_norm(K, uh_vem, predicted_displacements).item()

    return predicted_displacements, l2_error, energy_error, h1_error