import torch
from typing import Tuple, List

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
