import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    X: Input data (Batch, Time, InputDim)
    h_0: Initial hidden state (Batch, HiddenDim)
    W_xh: Input-to-hidden weights (HiddenDim, InputDim)
    W_hh: Hidden-to-hidden weights (HiddenDim, HiddenDim)
    b_h: Hidden bias (HiddenDim,)
    """
    batch_size, time_steps, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    
    # Initialize list to store hidden states for each time step
    h_all_list = []
    h_current = h_0
    
    # Loop over the time dimension (T)
    for t in range(time_steps):
        # Extract the input for the current time step x_t: (Batch, InputDim)
        x_t = X[:, t, :]
        
        # RNN Cell logic: h_t = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b_h)
        # Note: Weights are typically (Out, In), so we use transpose for @
        h_current = np.tanh(x_t @ W_xh.T + h_current @ W_hh.T + b_h)
        
        # Collect the new hidden state
        h_all_list.append(h_current)
    
    # Stack all collected hidden states along the time axis (axis 1)
    # Resulting shape: (Batch, Time, HiddenDim)
    h_all = np.stack(h_all_list, axis=1)
    
    # The final hidden state is the state at the last time step
    h_final = h_current
    
    return h_all, h_final