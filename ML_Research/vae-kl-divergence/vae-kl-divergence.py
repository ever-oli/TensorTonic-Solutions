import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    mu, log_var shapes: (Batch, LatentDim)
    """
    # 1. Compute variance from log-variance
    # sigma^2 = exp(log_var)
    var = np.exp(log_var)
    
    # 2. Apply the analytical KL formula element-wise:
    # kl_element = 1 + log(sigma^2) - mu^2 - sigma^2
    kl_element = 1 + log_var - np.square(mu) - var
    
    # 3. Sum over latent dimensions (axis 1) 
    # and multiply by -0.5
    batch_kl = -0.5 * np.sum(kl_element, axis=1)
    
    # 4. Return the mean divergence over the batch (axis 0)
    return float(np.mean(batch_kl))