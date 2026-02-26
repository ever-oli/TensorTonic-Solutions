import numpy as np

def conv_relu(x, out_channels):
    B, H, W, C = x.shape
    # Note: This helper provided in the snippet is a simplified 1x1 projection.
    # In a real VGG it would be a 3x3 spatial convolution.
    W_weights = np.random.randn(C, out_channels) * 0.1
    x = x @ W_weights
    return np.maximum(0, x)

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2,4))

def vgg_features(x: np.ndarray, config: list) -> np.ndarray:
    """
    Build VGG feature extractor from config.
    """
    out = x
    
    for layer in config:
        if isinstance(layer, int):
            # Apply 3x3 Conv + ReLU (using the provided helper)
            # The helper automatically handles the input channel tracking via x.shape
            out = conv_relu(out, layer)
        elif layer == 'M':
            # Apply 2x2 Max Pooling (stride 2)
            out = maxpool_2x2(out)
            
    return out