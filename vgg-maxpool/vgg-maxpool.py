import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    Input x shape: (Batch, H, W, C)
    Output shape: (Batch, H/2, W/2, C)
    """
    batch, h, w, c = x.shape
    
    # 1. Reshape to separate the pooling windows
    # We split H into (h // 2, 2) and W into (w // 2, 2)
    # The new shape is (Batch, H_out, 2, W_out, 2, C)
    reshaped_x = x.reshape(batch, h // 2, 2, w // 2, 2, c)
    
    # 2. Compute the maximum over the pooling axes (axis 2 and 4)
    # This reduces the 2x2 windows to a single maximum value
    out = reshaped_x.max(axis=(2, 4))
    
    return out