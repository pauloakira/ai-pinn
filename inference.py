import json
import torch
import numpy as np

import core.neural_backend as neural

def load_model(model):
    pass

def run_test(model, consolidated_file_path: str, geometry_file_path: str):
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

        predicted_displacements, l2_error, energy_error, h1_error, inference_time = neural.test_portic(
            nodes=nodes,
            material_params=material_params,
            model=model,
            uh_vem=uh_vem,
            concatanate=False,
            verbose=True
        )


if __name__ == "__main__":
    # Paths
    file_path = "data/consolidated_beam_test_data.json"
    geometry_path = "data/geometries/beam_64.json"

    # Run test pipeline
    run_test(consolidated_file_path=file_path,
            geometry_file_path=geometry_path)
