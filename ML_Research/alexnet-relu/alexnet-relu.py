import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: f(x) = max(0, x)"""
    # Use np.maximum for element-wise comparison against a scalar
    # This efficiently handles both scalars and multi-dimensional arrays
    return np.maximum(0, x)