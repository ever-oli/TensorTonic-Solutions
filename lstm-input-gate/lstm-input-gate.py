import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    
    # 1. Concatenate h_{t-1} and x_t along the feature dimension (axis=-1)
    # This combines the previous context with the current input
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute Input Gate (i_t)
    # Uses sigmoid activation to produce values in range [0, 1]
    i_t_linear = concat @ W_i.T + b_i
    i_t = sigmoid(i_t_linear)
    
    # 3. Compute Candidate Memory (C_tilde_t)
    # Uses tanh activation to produce values in range [-1, 1]
    c_tilde_linear = concat @ W_c.T + b_c
    c_tilde = np.tanh(c_tilde_linear)
    
    return i_t, c_tilde