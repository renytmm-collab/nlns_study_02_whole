import hashlib
import random
from contextlib import contextmanager
from .common import canonical

MASTER = 20261105
BOOTSTRAP = 20261106


def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical([MASTER, *parts]).encode()).digest()[:4], "big")


def local(*parts):
    return random.Random(seed(*parts))


@contextmanager
def stream(*parts):
    """Isolate legacy NumPy/Torch globals required by unmodified NLNS environment."""
    import numpy as np
    import torch
    py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    value = seed(*parts)
    try:
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        yield value
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(th_state)
