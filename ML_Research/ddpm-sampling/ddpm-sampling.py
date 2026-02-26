import numpy as np

def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM.
    """
    # 1. Initialize x_T with random Gaussian noise matching the desired output shape
    x_t = np.random.randn(*shape)
    
    # Pre-compute alphas and cumulative product (alpha_bar) to use inside the loop
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    
    # 2. Loop in reverse from T down to 1 (Hint 1)
    for t in range(T, 0, -1):
        # Predict the noise for the current timestep (Hint 2)
        # The model uses the current noisy image and the timestep to predict the noise
        epsilon_pred = model_predict(x_t, t)
        
        # Retrieve parameters for the current step (t is 1-indexed, so use t-1 for arrays)
        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]
        
        # Calculate the Posterior Mean (μ)
        # mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_pred)
        inv_sqrt_alpha_t = 1.0 / np.sqrt(alpha_t)
        noise_coeff = beta_t / np.sqrt(1.0 - alpha_bar_t)
        
        mu = inv_sqrt_alpha_t * (x_t - noise_coeff * epsilon_pred)
        
        # 3. Add stochastic noise if t > 1 (Hint 3)
        if t > 1:
            sigma_t = np.sqrt(beta_t)
            z = np.random.randn(*shape)
            x_t = mu + sigma_t * z
        else:
            # At t=1, do not add noise; the final step is deterministic
            x_t = mu
            
    # After the loop, the original noise has been fully transformed into x_0
    return x_t
