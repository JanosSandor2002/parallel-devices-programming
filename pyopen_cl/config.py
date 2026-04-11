# Konfiguráció
import os
N          = 10_000_000   # Tömb mérete
SEED       = 42
KERNEL_FILE = os.path.join(os.path.dirname(__file__), "sum_of_squares.cl")
RESULT_JSON = "results.json"