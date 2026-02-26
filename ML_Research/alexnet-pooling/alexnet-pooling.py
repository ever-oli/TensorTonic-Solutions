import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """Apply 2D max pooling (shape simulation)."""
    # 1. Extract input dimensions (Batch, Height, Width, Channels)
    batch_size, h_in, w_in, channels = x.shape
    
    # 2. Compute output spatial dimensions using the formula: 
    # H_out = floor((H_in - kernel) / stride) + 1
    # As noted in Hint 1: (55 - 3) / 2 + 1 = 27
    h_out = (h_in - kernel_size) // stride + 1
    w_out = (w_in - kernel_size) // stride + 1
    
    # 3. Create a representative output tensor of the correct shape
    # Pooling is applied independently per channel, so channel count remains the same
    output = np.zeros((batch_size, h_out, w_out, channels))
    
    return output