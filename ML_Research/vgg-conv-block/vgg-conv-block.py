import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block using NumPy.
    x shape: (Batch, Height, Width, Channels)
    """
    current_x = x
    
    for i in range(num_convs):
        in_channels = current_x.shape[-1]
        
        # 1. Initialize Weights (He initialization is standard for ReLU)
        # Shape: (kernel_h, kernel_w, in_channels, out_channels)
        limit = np.sqrt(2 / (3 * 3 * in_channels))
        weights = np.random.randn(3, 3, in_channels, out_channels) * limit
        bias = np.zeros(out_channels)
        
        # 2. Padding (Same padding for 3x3 kernel: pad 1 on all sides)
        # We pad Height (axis 1) and Width (axis 2)
        padded_x = np.pad(current_x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
        
        # 3. Convolutional Operation
        batch, h, w, _ = current_x.shape
        out = np.zeros((batch, h, w, out_channels))
        
        for i in range(3):
            for j in range(3):
                # Extract the 3x3 window shifted by (i, j) and multiply by weights
                # This is a vectorized approach for the sliding window
                window = padded_x[:, i:i+h, j:j+w, :]
                out += np.tensordot(window, weights[i, j], axes=([-1], [0]))
        
        out += bias
        
        # 4. ReLU Activation
        current_x = np.maximum(0, out)
        
    return current_x