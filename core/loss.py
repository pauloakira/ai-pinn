import torch
import numpy as np

def compute_loss(K, uh, F):
    """
    Compute the loss function as (K * uh - F)^2 using PyTorch

    Parameters:
    K (torch.Tensor): Stiffness matrix (ndof x ndof)
    uh (torch.Tensor): Solution vector (ndof x 1)
    F (torch.Tensor): Load vector (ndof x 1)

    Returns:
    torch.Tensor: The loss value
    """
    # Compute the residual
    R = torch.matmul(K, uh) - F
    
    # Compute the loss (squared residual)
    loss = torch.sum(R**2)
    
    return loss

def compute_loss_with_uh(uh_vem: np.array, uh: torch.Tensor):
    """
    Compute the loss function as (uh - uh_vem)^2 using PyTorch.

    Parameters:
    uh_vem (torch.Tensor): Solution vector from VEM (ndof x 1)
    uh (torch.Tensor): Solution vector (ndof x 1)

    Returns:
    torch.Tensor: The loss value
    """
    
    # Detach uh_vem if necessary to avoid tracking gradients
    uh_vem = torch.tensor(uh_vem, requires_grad=True)

    # Compute the loss (squared residual)
    loss = torch.sum((uh - uh_vem)**2)
    
    return loss

def compute_boundary_loss(uh, supp):
    """
    Compute the loss function for enforcing the Dirichlet boundary conditions.

    Parameters:
    uh (torch.Tensor): Solution vector (ndof x 1)
    supp (torch.Tensor): Support vector (ndof x N, where N is the number of nodes with boundary conditions)

    Returns:
    torch.Tensor: The loss value as a PyTorch tensor
    """
    # Initialize the loss as a scalar tensor with zero
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    for node in supp:
        if node[1] == 1:  # x-direction fixed
            k = 3 * int(node[0])
            loss = loss + uh[k] ** 2  # Avoid in-place operation by using a new tensor
        if node[2] == 1:  # y-direction fixed
            k = 3 * int(node[0]) + 1
            loss = loss + uh[k] ** 2  # Avoid in-place operation by using a new tensor
        if node[3] == 1:  # z-direction fixed
            k = 3 * int(node[0]) + 2
            loss = loss + uh[k] ** 2  # Avoid in-place operation by using a new tensor

    return loss

def compute_material_penalty(model, nodes, material_params_1, material_params_2, concatanate=False):
    """
    Computes a penalty for the model if the material parameters don't affect the predictions sufficiently.

    Parameters:
    mode (str): Material penalty mode ('l1' or 'l2')
    nodes (torch.Tensor): Node coordinates (nnodes x 3)
    material_params_1 (torch.Tensor): Material parameters for material 1 (nnodes x 1)
    material_params_2 (torch.Tensor): Material parameters for material 2 (nnodes x 1)

    Returns:
    torch.Tensor: Material penalty term as a scalar
    """
    if concatanate:
        input_1 = torch.cat((nodes, material_params_1))
        input_2 = torch.cat((nodes, material_params_2))

        uh_1 = model(input_1)
        uh_2 = model(input_2)
    else:
        uh_1 = model(nodes, material_params_1)
        uh_2 = model(nodes, material_params_2)
        

    penalty = torch.sum((uh_1 - uh_2) ** 2)
    return penalty

def normalize_loss_and_penalty(loss, material_penalty, beta=1e-4):
    """
    Normalizes the loss and material penalty by computing the scaling factor alpha.

    A factor beta is added to the loss and penalty to avoid division by zero.

    Parameters:
    loss (torch.Tensor): Loss term as a scalar
    material_penalty (torch.Tensor): Material penalty term as a scalar
    beta (float): Penalty factor (default: 1e-4)

    Returns:
    torch.Tensor: Normalized loss term as a scalar
    """
    loss_magnitude = loss.item() + beta
    penalty_magnitude = material_penalty.item() + beta

    alpha = loss_magnitude / penalty_magnitude
    
    return alpha


def compute_sobolev_loss(model, nodes, material_params, displacement_loss, concatanate=False):
    """
    Computes the Sobolev loss, includging both displacements and derivatives losses.

    Parameters:
    model (torch.nn.Module): Neural network model
    nodes (torch.Tensor): Node coordinates (nnodes x 3)
    material_params (torch.Tensor): Material parameters (nnodes x 1)
    uh_vem (torch.Tensor): Displacement field from VEM (ndof x 1)
    displacements_loss (torch.Tensor): Displacement loss as a scalar between uh and uh_vem

    Returns:
    torch.Tensor: Sobolev loss as a scalar
    """
    if concatanate:
        # Concatanete the nodes and material parameters
        input_vector = torch.cat((nodes, material_params))
        # Compute the displacement field from the neural network
        uh = model(input_vector)
    else:
        # Compute the displacement field from the neural network
        uh = model(nodes, material_params)

    # Compute the strain (first derivative)
    strain = torch.autograd.grad(uh, nodes, grad_outputs=torch.ones_like(uh), create_graph=True)[0]

    # Compute the curvature (second derivative)
    curvature = torch.autograd.grad(strain, nodes, grad_outputs=torch.ones_like(strain), create_graph=True)[0]

    # Compute the strain loss
    strain_loss = torch.sum(strain ** 2)

    # Compute the curvature loss
    curvature_loss = torch.sum(curvature ** 2)

    # Compute the total Sobolev loss
    sobolev_loss = displacement_loss + strain_loss + curvature_loss

    return sobolev_loss