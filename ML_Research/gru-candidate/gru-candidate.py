import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Compute candidate: h_tilde = tanh(W_h @ [r*h, x] + b_h)"""
    
    # 1. Apply reset gate to previous hidden state (Element-wise multiplication)
    # This filters the past history based on the reset gate's decision
    gated_h = r_t * h_prev
    
    # 2. Concatenate gated history with current input
    # Shape becomes (Batch, HiddenDim + InputDim)
    concat = np.concatenate([gated_h, x_t], axis=-1)
    
    # 3. Apply linear transformation and tanh activation
    # Weights are (HiddenDim, HiddenDim + InputDim), so we use .T for batch multiplication
    linear_transform = concat @ W_h.T + b_h
    h_tilde = np.tanh(linear_transform)
    
    return h_tilde