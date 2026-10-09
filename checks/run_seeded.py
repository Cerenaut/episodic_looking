"""
Run a script unchanged after seeding python, numpy and torch globally, so an unseeded script (e.g. an old commit's)
can be compared byte for byte with a new one. Usage, from the code root to run:
    python <path>/checks/run_seeded.py SEED script.py [script args...]
"""
import random
import runpy
import sys

if __name__ == "__main__":
    import numpy as np
    import torch
    seed = int(sys.argv[1])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    sys.path.insert(0, ".")
    script = sys.argv[2]
    sys.argv = sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
