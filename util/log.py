import os
from datetime import datetime


def periodic(t, period):
    return (t % period) == 0

def loss_to_float_with_norm(x, n = None) -> float:
    if not isinstance(x, float):
        x = x.item()
    if n is not None:
        return x / float(n)
    return x

def tensor_to_float_with_norm(t, n = None) -> float:
    if n is not None:
        return float(t.detach().sum()) / float(n)
    return float(t.detach().sum())

def get_run_path(prefix:str, path:str = "./runs"):
    """
    Returns a nice folder structure to organise runs
    """
    now = datetime.now()
    suffix = now.strftime("%Y/%m/%d/%H-%M-%S")
    run_path = os.path.join(path, prefix, suffix)
    return run_path

def create_run_path(path:str):
    try: 
        os.makedirs(path)
    except FileExistsError:
        pass


