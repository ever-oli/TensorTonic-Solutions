import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)"""
    
    # 1. Concatenate the previous hidden state and current input
    # If h_prev is (N, H) and x_t is (N, D), concat is (N, H + D)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Linear transformation: concat @ W_z.T + b_z
    # Weights are typically (HiddenDim, HiddenDim + InputDim)
    linear_transform = concat @ W_z.T + b_z
    
    # 3. Apply sigmoid activation
    # This squashes the values into the [0, 1] range
    z_t = sigmoid(linear_transform)
    
    return z_t