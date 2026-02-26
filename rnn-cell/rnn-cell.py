import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    
    x_t: Current input (Batch, InputDim)
    h_prev: Previous hidden state (Batch, HiddenDim)
    W_xh: Input-to-hidden weights (HiddenDim, InputDim)
    W_hh: Hidden-to-hidden weights (HiddenDim, HiddenDim)
    b_h: Hidden bias (HiddenDim,)
    """
    # 1. Linear transformation of current input: x_t @ W_xh.T
    input_term = x_t @ W_xh.T
    
    # 2. Linear transformation of previous hidden state: h_prev @ W_hh.T
    hidden_term = h_prev @ W_hh.T
    
    # 3. Sum terms, add bias, and apply tanh activation
    # h_t = tanh(input_term + hidden_term + b_h)
    h_t = np.tanh(input_term + hidden_term + b_h)
    
    return h_t