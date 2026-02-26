import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    
    # 1. Concatenate previous hidden state and current input
    # Shape: (Batch, HiddenDim + InputDim)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # 2. Compute all four gates using the concatenated input
    # Forget Gate (f_t): Sigmoid activation
    f_t = sigmoid(concat @ W_f.T + b_f)
    
    # Input Gate (i_t): Sigmoid activation
    i_t = sigmoid(concat @ W_i.T + b_i)
    
    # Candidate Memory (C_tilde_t): Tanh activation
    c_tilde = np.tanh(concat @ W_c.T + b_c)
    
    # Output Gate (o_t): Sigmoid activation
    o_t = sigmoid(concat @ W_o.T + b_o)
    
    # 3. Update Cell State (C_t)
    # Combine old memory (filtered by forget gate) and new info (filtered by input gate)
    C_t = f_t * C_prev + i_t * c_tilde
    
    # 4. Compute Hidden State (h_t)
    # The output is a filtered version of the updated cell state
    h_t = o_t * np.tanh(C_t)
    
    return h_t, C_t