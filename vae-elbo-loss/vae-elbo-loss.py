import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    x: original input
    x_recon: reconstructed output from decoder
    mu: mean from encoder
    log_var: log-variance from encoder
    """
    # 1. Compute Reconstruction Loss (MSE/BCE)
    # The requirement specifies summing over features and mean over batch
    # Using MSE as shown in the example: (x - x_recon)^2
    recon_loss_per_sample = np.sum(np.square(x - x_recon), axis=1)
    recon_loss = np.mean(recon_loss_per_sample)
    
    # 2. Compute KL Divergence
    # Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    var = np.exp(log_var)
    kl_per_sample = -0.5 * np.sum(1 + log_var - np.square(mu) - var, axis=1)
    kl_loss = np.mean(kl_per_sample)
    
    # 3. Total Loss
    total_loss = recon_loss + kl_loss
    
    return {
        "total": float(total_loss),
        "recon": float(recon_loss),
        "kl": float(kl_loss)
    }