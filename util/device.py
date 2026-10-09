import os
import random

import numpy as np
import torch


def get_device():
    """
    Use this function to explicitly try to store all tensors on the device. It assumes a 
    single GPU device. It will fall back to CPU if no GPU is available, e.g. for local 
    debugging.

    Priority: cuda > mps > cpu. Set the environment variable EPISODIC_DEVICE (e.g. "cpu",
    "mps", "cuda") to override the automatic choice without changing any code.
    """
    override = os.environ.get("EPISODIC_DEVICE")
    if override:
        return torch.device(override)

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


def seed_all(seed:int):
    """
    Seed python's random, numpy's global generator and torch on every available device (CPU, MPS, CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
