import os
import json
import torch
import numpy as np

import core.neural_backend as neural

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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend is available!")
    else:
        device = torch.device("cpu")
    print("MPS backend is not available. Using CPU.")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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

    # Material parameters
    E = loaded_dataset[0]['E']
    A = loaded_dataset[0]['A']
    I = loaded_dataset[0]['I']
    E_dist = loaded_dataset[0]['E_dist']
    A_dist = loaded_dataset[0]['A_dist']
    I_dist = loaded_dataset[0]['I_dist']

    # Cast to tensor
    material_params_1 = torch.tensor([E, A, I], dtype=torch.float32)
    material_params_2 = torch.tensor([E_dist, A_dist, I_dist], dtype=torch.float32)

    # Normalizing
    nodes, material_params_1 = neural.normalize_inputs(nodes, material_params_1)
    _, material_params_2 = neural.normalize_inputs(nodes, material_params_2) 

    # Casting nodes
    nodes = nodes.flatten()
    nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)

    # Setting the VEM solution
    uh_vem = np.array(loaded_dataset[0]['displacements'])

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
    loaded_model.load_state_dict(torch.load("data/models/neural_vem_64.pth"))

    # Set the model to evaluation mode (important for inference)
    loaded_model.eval()

    # Test the model using the new material parameters
    predicted_displacements, l2_error, energy_error, h1_error, inference_time = neural.test_portic(
        nodes=nodes,
        material_params=material_params_1,
        model=loaded_model,  # Use the loaded model for inference
        uh_vem=uh_vem,
        concatanate=False,
        verbose=True
    )


if __name__ == "__main__":

    type = "portic"

    # Paths for beam 
    file_path = "data/consolidated_beam_test_data.json"
    geometry_path = "data/geometries/beam_64.json"
    model_path = "data/models/neural_vem_beam_64.pth"

    # Paths for portic
    file_path = "data/consolidated_portic_test_data.json"
    geometry_path = "data/geometries/portic_64.json"
    model_path = "data/models/neural_vem_64.pth" 

    # Run test pipeline
    # run_test(model_path=model_path,
    #    consolidated_file_path=file_path,
    #    geometry_file_path=geometry_path,
    #    type=type)
    test(consolidated_file_path=file_path, geometry_file_path=geometry_path)