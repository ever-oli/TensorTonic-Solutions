import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule from beta_1 to beta_T.
    """
    # Create linearly spaced values between beta_1 and beta_T
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    """
    # Timesteps from 1 to T
    t = np.arange(1, T + 1)
    
    # Calculate f(0) for the denominator
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    
    # Calculate f(t) for the numerator
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    
    # alpha_bar_t = f(t) / f(0)
    alpha_bars = f_t / f_0
    
    return alpha_bars

def alpha_bar_to_betas(alpha_bars: np.ndarray) -> np.ndarray:
    """
    Convert alpha_bar schedule to beta schedule.
    """
    # alpha_bar_{t-1}: shift alpha_bars to the right by 1, and pad with 1.0 at t=0
    alpha_bars_prev = np.concatenate(([1.0], alpha_bars[:-1]))
    
    # Since alpha_bar_t = alpha_bar_{t-1} * (1 - beta_t), we can solve for beta_t:
    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1.0 - (alpha_bars / alpha_bars_prev)
    
    # Clip betas to be strictly less than 1 to avoid numerical singularities (variance becoming 0)
    # 0.999 is standard practice in DDPM implementations
    return np.clip(betas, 0.0, 0.999)