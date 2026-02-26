import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    
    dh_next: Gradient of loss w.r.t. current hidden state h_t (Batch, HiddenDim)
    h_t: Current hidden state (Batch, HiddenDim)
    h_prev: Previous hidden state h_{t-1} (Batch, HiddenDim)
    x_t: Current input (Batch, InputDim)
    W_hh: Hidden-to-hidden weights (HiddenDim, HiddenDim)
    """
    
    # 1. Compute the pre-activation gradient (through the tanh)
    # Derivative of tanh(z) is (1 - tanh^2(z)). Since h_t = tanh(z), 
    # the derivative is (1 - h_t^2).
    dtanh = (1 - np.square(h_t)) * dh_next
    
    # 2. Compute the gradient w.r.t. the recurrent weights W_hh
    # Using Hint 2: dW_hh = dtanh.T @ h_prev
    # This sums the gradients over the batch dimension.
    dW_hh = dtanh.T @ h_prev
    
    # 3. Compute the gradient to pass back to the previous hidden state h_{t-1}
    # Using Hint 3: dh_prev = dtanh @ W_hh
    dh_prev = dtanh @ W_hh
    
    return dh_prev, dW_hh