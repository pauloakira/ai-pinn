import os
import json
import torch
import numpy as np

import core.neural_backend as neural
from utils.helpers import consolidate_json_in_dataset

def load_model(model_path: str, ndof: int, input_dim_nodes: int, input_dim_materials: int):
    # Layers definition
    nodes_layers = [128, 256, 512, 512, 512, 512]  # Layers for nodes sub-network
    material_layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 4096, 4096, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512] # Layers for materials sub-network
    final_layers = [1024, 1024, 1024, 1024, 1024, 1024] # Layers for final combination network

    # Create a new instance of the model (with the same architecture)
    loaded_model = neural.BeamApproximatorWithMaterials(
        input_dim_nodes=input_dim_nodes, 
        input_dim_materials=input_dim_materials, 
        nodes_layers=nodes_layers, 
        material_layers=material_layers, 
        final_layers=final_layers, 
        ndof=ndof
    )

    # Load the saved model state
    loaded_model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important for inference)
    loaded_model.eval()

    return loaded_model


def run_test(model_path: str, consolidated_file_path: str, geometry_file_path: str, type: str = "beam"):
    # Load test dataset
    with open(consolidated_file_path, 'r') as file:
        dataset = json.load(file)
    loaded_dataset = dataset['dataset']

    # Load geometry
    with open(geometry_file_path, 'r') as file:
        geometry = json.load(file)

    # Process nodes
    nodes = np.array(geometry['nodes'])
    nodes = nodes.flatten()
    nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)

    # Number of degrees of freedom
    if type == "beam":
        ndof = len(nodes)
        print(f"Ndof :: {ndof}")
        input_dim_nodes = len(nodes)
        input_dim_materials = 3
    elif type == "portic":
        ndof = 3*len(nodes)
        input_dim_nodes = 2*len(nodes)
        input_dim_materials = 3
    else:
        raise("Invalid type.")

    for data in loaded_dataset:
        # Consolidate material parameters
        E = data['E']
        I = data['I']
        A = data['A']
        material_params = torch.tensor([E, A, I], dtype=torch.float32)

        # Casting to tensor
        uh_vem = np.array(data['displacements'])
        uh_vem = torch.tensor(uh_vem, dtype=torch.float32, requires_grad=True)

        # Normalizing
        nodes, material_params = neural.normalize_inputs(nodes, material_params)

        model = load_model(model_path, 
                        ndof=ndof, 
                        input_dim_nodes=input_dim_nodes, 
                        input_dim_materials=input_dim_materials)

        predicted_displacements, l2_error, energy_error, h1_error, inference_time = neural.test_portic(
            nodes=nodes,
            material_params=material_params,
            model=model,
            uh_vem=uh_vem,
            concatanate=False,
            verbose=True
        )

        print(h1_error)

def test(consolidated_file_path: str, geometry_file_path: str,):
    # Load test dataset
    with open(consolidated_file_path, 'r') as file:
        dataset = json.load(file)
    loaded_dataset = dataset['dataset']

    # Load geometry
    with open(geometry_file_path, 'r') as file:
        geometry = json.load(file)

    nodes = np.array(geometry['nodes']) 

    # Parameters
    ndof = 3 * len(nodes)
    input_dim_nodes = 2*len(nodes)
    input_dim_materials = 3

    # H1 values
    h1_values = []

    # Layers definition
    nodes_layers = [128, 256, 512, 512, 512, 512]  # Layers for nodes sub-network
    material_layers = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 4096, 4096, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512] # Layers for materials sub-network
    final_layers = [1024, 1024, 1024, 1024, 1024, 1024] # Layers for final combination network

    # Create a new instance of the model (with the same architecture)
    loaded_model = neural.BeamApproximatorWithMaterials(
        input_dim_nodes=input_dim_nodes, 
        input_dim_materials=input_dim_materials, 
        nodes_layers=nodes_layers, 
        material_layers=material_layers, 
        final_layers=final_layers, 
        ndof=ndof
    )

    # Load the saved model state
    loaded_model.load_state_dict(torch.load("data/models/neural_vem_8.pth"))

    # Set the model to evaluation mode (important for inference)
    loaded_model.eval()

    for data in loaded_dataset:

        # Material parameters
        E = data['E']
        A = data['A']
        I = data['I']

        # Cast to tensor
        material_params = torch.tensor([E, A, I], dtype=torch.float32)

        # Normalizing
        nodes, material_params = neural.normalize_inputs(nodes, material_params)

        # Casting nodes
        nodes = nodes.flatten()
        nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)

        # Setting the VEM solution
        uh_vem = np.array(data['displacements'])        

        # Test the model using the new material parameters
        predicted_displacements, l2_error, energy_error, h1_error, inference_time = neural.test_portic(
            nodes=nodes,
            material_params=material_params,
            model=loaded_model,  # Use the loaded model for inference
            uh_vem=uh_vem,
            concatanate=False,
            verbose=True
        )

        h1_values.append(h1_error)

    h1_values = np.array(h1_values)
    mean = np.mean(h1_values)
    std = np.std(h1_values)

    print("----------------------------------------")
    print(f"Mean :: {mean}")
    print(f"Std :: {std}")
    print("----------------------------------------")

def generate_test_dataset(num_elements_per_edge: int):
    directory_path_name = f"data/datasets/portic_test/data_{num_elements_per_edge}/"
    consolidate_json_in_dataset(directory_path_name, f"consolidated_portic_test_data_{num_elements_per_edge}.json")


if __name__ == "__main__":

    type = "portic"
    num_elements_per_edge = 8

    # Paths for beam 
    # file_path = "data/consolidated_beam_test_data.json"
    # geometry_path = "data/geometries/beam_64.json"
    # model_path = "data/models/neural_vem_beam_64.pth"

    generate_test_dataset(num_elements_per_edge=num_elements_per_edge)

    # Paths for portic
    file_path = f"data/consolidated_portic_test_data_{num_elements_per_edge}.json"
    geometry_path = f"data/geometries/portic_{num_elements_per_edge}.json"
    model_path = f"data/models/neural_vem_{num_elements_per_edge}.pth" 

    # Run test pipeline
    # run_test(model_path=model_path,
    #    consolidated_file_path=file_path,
    #    geometry_file_path=geometry_path,
    #    type=type)
    # generate_test_dataset(num_elements_per_edge=128)
    test(consolidated_file_path=file_path, geometry_file_path=geometry_path)
