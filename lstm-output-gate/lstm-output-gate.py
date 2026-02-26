import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    
    # 1. Concatenate the previous hidden state and current input
    # Shape becomes (Batch, HiddenDim + InputDim)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute the Output Gate (o_t)
    # Uses sigmoid to ensure values are in the range [0, 1]
    o_t_linear = concat @ W_o.T + b_o
    o_t = sigmoid(o_t_linear)
    
    # 3. Compute the Hidden State (h_t)
    # Filter the updated cell state C_t through a tanh and the output gate
    h_t = o_t * np.tanh(C_t)
    
    return o_t, h_t