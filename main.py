import os
import torch
import core.grad_norm as gn
import core.neural_backend as neural
from utils.datasets import generate_beam_dataset_from_json
from utils.helpers import generate_beam_parameters, consolidate_json_in_dataset
import utils.mesh as mesh
from utils.utils import retrieve_device

# Setup the device
device = retrieve_device()

# Define the number of elements per edge
num_elements_per_edge = 128

directory_path_name = f"data/datasets/portic/data_{num_elements_per_edge}_order_5/"
consolidate_json_in_dataset(directory_path_name, f"consolidated_portic_data_{num_elements_per_edge}_order_5.json")

# Filename
result_filename = f"data/consolidated_portic_data_{num_elements_per_edge}_order_5.json"
geometry_filename = f"data/geometries/portic_{num_elements_per_edge}.json"

# geometry data
L = 2.0
I = 1e-4
A = 1

# material data
E = 27e6

# Define load parameters
q = -400
t = 0

# Generate the geometry
# nodes, elements, supp, load = mesh.generate_portic_geometry(num_elements_per_edge, L)
dataset = generate_beam_dataset_from_json(result_filename=result_filename, geometry_filename=geometry_filename)
nodes = dataset[0]['nodes']


# Hyperparameters
num_epochs = 80
concatenate = False

nodes_layers = [128, 256, 512, 512, 512, 512]  # Layers for nodes sub-network
material_layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 4096, 4096, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512] # Layers for materials sub-network
final_layers = [1024, 1024, 1024, 1024, 1024, 1024] # Layers for final combination network

model, total_loss_values, loss_values, material_loss_values, sobolev_loss_values, alpha_values_values = neural.train_with_few_materials(
    nodes=nodes,
    epochs=num_epochs,
    nodes_layers=nodes_layers,
    material_layers=material_layers,
    final_layers=final_layers,
    device=device,
    number_of_materials=32,
    from_json=True,
    result_filename=result_filename,
    geometry_filename=geometry_filename,
    type = neural.StructureType.Portic
    )

# Save the trained model
os.makedirs("data/models", exist_ok=True)
torch.save(model.state_dict(), f"data/models/neural_vem_{num_elements_per_edge}_order_5.pth")