import torch


def get_device():
    """
    Use this function to explicitly try to store all tensors on the device. It assumes a 
    single GPU device. It will fall back to CPU if no GPU is available, e.g. for local 
    debugging.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def set_default_device(device):
    """
    Call this at the start of the program to ensure all tensors are initialized on 
    this device without being explicit everywhere. This works reliably if the entire 
    model and data fits on one device.

    This can be combined with `get_device` to default all tensors to the GPU if available.

    Note: from_numpy() will still always be CPU and so should be followed with .to(device)
    """
    torch.set_default_device(device)