import torch
from dataclasses import dataclass

import solve_vem
from utils import mesh
from utils.helpers import generate_beam_parameters

@dataclass
class BeamInformation:
    number_elements_per_edge: int
    length: float
    vertical_load: float
    horizontal_load: float


def z_score_normalize(value, mean_val, std_val):
    """Z-score normalization.
    
    Parameters:
    value (float): Value to normalize
    mean_val (float): Mean value
    std_val (float): Standard deviation value

    Returns:
    float: Normalized value

    """
    return (value - mean_val) / std_val


def generate_beam_dataset(elastic_module_range: list, inertia_moment_range: list, area_range: list, num_samples: int,
                          beam_information: BeamInformation = BeamInformation(8, 2, -400, 0)):
    """
    Function to generate a dataset of beam parameters with z-score normalized material properties.

    Parameters:
    beam_information (BeamInformation): Beam information
    elastic_module_range (list): Range of elastic modulus values
    inertia_moment_range (list): Range of inertia moment values
    area_range (list): Range of area values
    num_samples (int): Number of samples to generate

    Returns:
    list: List of dictionaries containing the dataset
    """
    # Initialize lists to store the dataset
    dataset = []

    # Generate the material parameters
    params = generate_beam_parameters(elastic_module_range, inertia_moment_range, area_range, num_samples)

    # Extract E, I, and A values for normalization
    E_values = [param['E'] for param in params]
    I_values = [param['I'] for param in params]
    A_values = [param['A'] for param in params]

    # Calculate mean and standard deviation
    E_mean, E_std = torch.mean(torch.tensor(E_values)), torch.std(torch.tensor(E_values))
    I_mean, I_std = torch.mean(torch.tensor(I_values)), torch.std(torch.tensor(I_values))
    A_mean, A_std = torch.mean(torch.tensor(A_values)), torch.std(torch.tensor(A_values))

    # Normalize each parameter using z-score normalization
    for i in range(num_samples):
        E, I, A = params[i]['E'], params[i]['I'], params[i]['A']

        # Z-score normalization
        E_norm = z_score_normalize(E, E_mean, E_std)
        I_norm = z_score_normalize(I, I_mean, I_std)
        A_norm = z_score_normalize(A, A_mean, A_std)

        # Generate the geometry
        nodes, elements, supp, load = mesh.generate_portic_geometry(beam_information.number_elements_per_edge, 
                                                                    beam_information.length)

        # Solve the problem using the VEM
        uh_vem, K, f, _ = solve_vem.solve_1d(nodes, 
                                            elements, 
                                            supp, 
                                            E, 
                                            A, 
                                            I, 
                                            load, 
                                            beam_information.vertical_load, 
                                            beam_information.horizontal_load, 
                                            verbose=False)

        # Convert nodes to tensor
        nodes = nodes.flatten()
        nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)

        # Store the dataset
        dataset.append({
            "nodes": nodes,
            "elements": elements,
            "supp": supp,
            "load": load,
            "uh_vem": uh_vem,
            "K": K,
            "f": f,
            "material_params": torch.tensor([E_norm, A_norm, I_norm], dtype=torch.float32),
            "distorted_material_params": torch.tensor([z_score_normalize(E * 1.3, E_mean, E_std), 
                                                      z_score_normalize(A * 1.1, A_mean, A_std), 
                                                      z_score_normalize(I * 0.3, I_mean, I_std)], dtype=torch.float32)
        })

    return dataset