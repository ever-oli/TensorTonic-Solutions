import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)"""
    
    # 1. Concatenate h_prev and x_t along the last axis
    # Shape becomes (Batch, HiddenDim + InputDim)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Apply linear transformation: concat @ W_r.T + b_r
    # Weights are (HiddenDim, HiddenDim + InputDim), so we use .T for batch multiplication
    linear_transform = concat @ W_r.T + b_r
    
    # 3. Apply sigmoid activation to get gate values in range [0, 1]
    r_t = sigmoid(linear_transform)
    
    return r_t