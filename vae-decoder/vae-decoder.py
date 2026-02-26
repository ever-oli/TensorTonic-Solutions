import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    z shape: (Batch, LatentDim)
    returns: x_hat shape (Batch, output_dim)
    """
    batch_size, latent_dim = z.shape
    hidden_dim = 256  # Hidden layer size to mirror the encoder
    
    # 1. Hidden Layer: Linear + ReLU
    # Initialize with small random values for stability
    w_h = np.random.randn(latent_dim, hidden_dim) * 0.01
    b_h = np.zeros(hidden_dim)
    h = np.maximum(0, z @ w_h + b_h)
    
    # 2. Output Layer: Linear + Sigmoid
    # Projects the hidden representation back to the original data dimension
    w_out = np.random.randn(hidden_dim, output_dim) * 0.01
    b_out = np.zeros(output_dim)
    logits = h @ w_out + b_out
    
    # Sigmoid activation ensures output is in range [0, 1] (common for normalized data)
    x_hat = 1 / (1 + np.exp(-logits))
    
    return x_hat