import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    z = mu + std * epsilon
    """
    # 1. Compute standard deviation from log-variance
    # std = exp(0.5 * log_var)
    std = np.exp(0.5 * log_var)
    
    # 2. Sample noise epsilon from a standard normal distribution N(0, I)
    # The shape must match the input mu
    epsilon = np.random.randn(*mu.shape)
    
    # 3. Scale and shift the noise: z = mu + sigma * epsilon
    z = mu + std * epsilon
    
    return z