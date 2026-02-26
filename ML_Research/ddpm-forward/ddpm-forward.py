import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    # 1. Compute alpha_t = 1 - beta_t
    alphas = 1.0 - betas
    
    # 2. Compute the cumulative product (alpha_bar)
    # alpha_bar_t = product_{i=1 to t} alpha_i
    alpha_bar = np.cumprod(alphas, axis=0)
    
    return alpha_bar

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0).
    """
    # 3. Retrieve the correct alpha_bar_t value (Hint 3)
    # We use t-1 because array indices are 0-based while timesteps are 1-based
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]
    
    # 4. Sample Gaussian noise epsilon matching x_0 shape (Hint 2)
    epsilon = np.random.randn(*x_0.shape)
    
    # 5. Compute x_t using the closed-form formula (Hint 1)
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
    
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    
    return x_t, epsilon