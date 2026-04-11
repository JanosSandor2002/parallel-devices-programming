# data
import numpy as np
from config import N, SEED

def generate_data():
    rng = np.random.default_rng(SEED)
    x = rng.random(N).astype(np.float32)
    return x