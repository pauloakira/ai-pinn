import os
import torch

def retrieve_device(set_cpu: bool = False):
    """
    Retrieve the device for the model.

    Returns:
    torch.device: Device for the model
    """
    if set_cpu:
        device = torch.device("cpu")
        print("Using CPU.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend is available!")
    else:
        device = torch.device("cpu")
        print("MPS backend is not available. Using CPU.")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    return device