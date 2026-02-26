import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """Apply Local Response Normalization across channels."""
    # 1. Get dimensions
    # x shape: (Batch, Height, Width, Channels)
    batch_size, h, w, c = x.shape
    
    # 2. Square the input activations
    squared_x = np.square(x)
    
    # 3. Create a sliding window sum over the channel axis
    # We pad the channel dimension to handle the edges of the window
    pad = n // 2
    # Pad only the last axis (channels)
    padded_sq = np.pad(squared_x, ((0, 0), (0, 0), (0, 0), (pad, pad)), mode='constant')
    
    # Accumulate the sum of squares within the window [i - n/2, i + n/2]
    sum_sq = np.zeros_like(x)
    for i in range(n):
        # Sliding window sum using slices
        sum_sq += padded_sq[:, :, :, i:i+c]
        
    # 4. Compute the normalization factor: (k + alpha * sum_sq) ^ beta
    scale = (k + alpha * sum_sq) ** beta
    
    # 5. Divide the original input by the scale factor
    return x / scale