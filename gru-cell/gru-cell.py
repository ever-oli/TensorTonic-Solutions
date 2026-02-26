import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Complete GRU cell forward pass."""
    
    # 1. Concatenate previous hidden state and current input for gates
    # Shape: (Batch, HiddenDim + InputDim)
    concat_gates = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute Reset Gate (r_t) and Update Gate (z_t)
    # Using sigmoid activation to keep values in [0, 1]
    r_t = sigmoid(concat_gates @ W_r.T + b_r)
    z_t = sigmoid(concat_gates @ W_z.T + b_z)
    
    # 3. Compute Candidate Hidden State (h_tilde_t)
    # Apply reset gate to h_prev before concatenating with x_t
    gated_h = r_t * h_prev
    concat_cand = np.concatenate([gated_h, x_t], axis=-1)
    h_tilde = np.tanh(concat_cand @ W_h.T + b_h)
    
    # 4. Compute Final Hidden State (h_t)
    # Linear interpolation between h_prev and h_tilde controlled by z_t
    h_t = z_t * h_prev + (1 - z_t) * h_tilde
    
    return h_t