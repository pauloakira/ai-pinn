import torch
import numpy as np
from typing import List

def calculate_gradient_norm(model, losses: List[torch.Tensor]):
    """
    Calculate the gradient norms for each loss function.

    G_W^i(t) => notation used in the original paper.

    Parameters:
    model (nn.Module): Neural network model
    losses (List[List[float]]): List of losses for each epoch and iteration

    Returns:
    List[List[float]]: Gradient norms for each loss function
    """

    # Ensure that gradients are zero
    model.zero_grad()

    # Initialize list to store gradient norms
    grad_norms = []

    # Get the shared weights from the shared layer (final layer)
    shared_weights_in = model.final_in.weight
    shared_weights_hidden = [layer.weight for layer in model.final_hidden]
    shared_weights_out = model.final_out.weight

    # Iterate through each task's weighted loss and calculate gradients
    for loss in losses:
        # Perform the backward pass to compute the necessary gradients
        loss.backward(retain_graph=True)

        # Collect the gradients of shared weights
        shared_gradients = []

        # Calculate the gradient norms for the shared weights
        if shared_weights_in.grad is not None:
            shared_gradients.append(shared_weights_in.grad.view(-1))
        
        for weight in shared_weights_hidden:
            if weight.grad is not None:
                shared_gradients.append(weight.grad.view(-1))
        
        if shared_weights_out.grad is not None:
            shared_gradients.append(shared_weights_out.grad.view(-1))

        # Concatenate the gradients and compute the L2 norm
        all_shared_gradients = torch.cat(shared_gradients)
        grad_norm = torch.norm(all_shared_gradients, p=2)

        # Append the gradient norm to the list
        grad_norms.append(grad_norm.item())

        # Zero out the gradients for the next iteration
        model.zero_grad()

    return grad_norms


def compute_avg_norm(grad_norms: List[float]):
    """
    Compute the average gradient norm for each loss function.

    Parameters:
    grad_norms (List[List[float]]): List of gradient norms for each loss function

    Returns:
    List[float]: Average gradient norm for each loss function
    """
    return np.mean(grad_norms, axis=0)


def compute_loss_ratio(current_loss_value: float, initial_loss_value: float)-> float:
    """
    Compute the loss ratio between the current loss value and the initial loss value.

    Parameters:
    current_loss_value (float): Current loss value
    initial_loss_value (float): Initial loss value

    Returns:
    float: Loss ratio
    """
    return current_loss_value / (initial_loss_value)


def compute_relative_inverse_training_rate(loss_ratios: List[float])-> List[float]:
    """
    Calculate the relative inverse training rate for each loss ratio.

    Parameters:
    loss_ratios (List[float]): List of loss ratios

    Returns:
    List[float]: Relative inverse training rate for each loss ratio
    """
    avg_loss_ratio = np.mean(loss_ratios)
    return [loss_ratio / avg_loss_ratio for loss_ratio in loss_ratios]

def compute_grad_norm_loss(grad_norms: List[float], loss_ratios: List[float], alpha: float = 0.5)-> torch.Tensor:
    """
    Compute the gradient norm loss for each loss function.

    Parameters:
    grad_norms (List[float]): List of gradient norms for each loss function
    loss_ratios (List[float]): List of loss ratios

    Returns:
    List[float]: Gradient norm loss for each loss function
    """
    # Setup the number of tasks
    num_tasks = len(grad_norms)

    # Convert grad_norms to tensors
    grad_norms = torch.tensor(grad_norms)
    
    # Calculate the average gradient norm across all tasks (convert to tensor)
    avg_grad_norm = torch.tensor(compute_avg_norm(grad_norms.tolist()))

    # Compute the relative inverse training rate for each task
    ri = compute_relative_inverse_training_rate(loss_ratios)

    loss_grad = torch.tensor(0.0, requires_grad=True)

    for i in range(num_tasks):
        ri_tensor = torch.tensor(ri[i])
        loss_task_grad = torch.abs(grad_norms[i] - avg_grad_norm * torch.pow(ri_tensor, alpha))
        loss_grad = loss_grad + loss_task_grad

    return loss_grad 
