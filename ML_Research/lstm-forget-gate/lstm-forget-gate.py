import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    
    # 1. Concatenate h_{t-1} and x_t along the feature dimension (axis=-1)
    # If h is (N, H) and x is (N, D), concat is (N, H + D)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Linear transformation: concat @ W_f.T + b_f
    # Since W_f is typically (Hidden, H + D), we transpose it for the dot product
    linear_transform = concat @ W_f.T + b_f
    
    # 3. Apply sigmoid activation to get values in range [0, 1]
    f_t = sigmoid(linear_transform)
    
    return f_t