import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.
    """
    # Create a zero-initialized array with shape (batch_size, hidden_dim)
    # The default data type for np.zeros is float
    h0 = np.zeros((batch_size, hidden_dim))
    
    return h0