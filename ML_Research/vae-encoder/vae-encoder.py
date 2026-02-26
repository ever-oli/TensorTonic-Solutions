import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    x shape: (Batch, InputDim)
    returns: (mu, log_var) both of shape (Batch, latent_dim)
    """
    batch_size, input_dim = x.shape
    hidden_dim = 256  # Example hidden dimension
    
    # 1. Hidden Layer (Linear + ReLU)
    # Initialize weights with small random values for stability
    w_h = np.random.randn(input_dim, hidden_dim) * 0.01
    b_h = np.zeros(hidden_dim)
    h = np.maximum(0, x @ w_h + b_h)
    
    # 2. Output Layer for Mean (mu)
    # Mapping hidden representation to latent_dim
    w_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_mu = np.zeros(latent_dim)
    mu = h @ w_mu + b_mu
    
    # 3. Output Layer for Log-Variance (log_var)
    # Separate weight matrix for the second distribution parameter
    w_log_var = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_log_var = np.zeros(latent_dim)
    log_var = h @ w_log_var + b_log_var
    
    return mu, log_var