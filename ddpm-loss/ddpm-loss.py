import numpy as np

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.
    """
    batch_size = x_0.shape[0]
    
    # 1. Sample a random timestep t in [1, T] for each batch element (Hint 2)
    # np.random.randint bounds are [low, high), so we use T + 1
    t = np.random.randint(1, T + 1, size=(batch_size,))
    
    # 2. Compute alpha_bar (cumulative product of 1 - betas)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    
    # Get alpha_bar_t for the sampled timesteps (t is 1-indexed, so we use t-1)
    a_bar_t = alpha_bars[t - 1]
    
    # Reshape a_bar_t so it can broadcast correctly with x_0 (e.g., from (B,) to (B, 1, 1, 1))
    broadcast_shape = [-1] + [1] * (x_0.ndim - 1)
    a_bar_t = a_bar_t.reshape(broadcast_shape)
    
    # 3. Sample random true noise epsilon matching the shape of x_0
    epsilon = np.random.randn(*x_0.shape)
    
    # 4. Generate x_t using the forward diffusion closed-form formula (Hint 3)
    # x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * epsilon
    x_t = np.sqrt(a_bar_t) * x_0 + np.sqrt(1.0 - a_bar_t) * epsilon
    
    # 5. Get the model's predicted noise
    epsilon_pred = model_predict(x_t, t)
    
    # 6. Compute the simplified loss: MSE between true noise and predicted noise (Hint 1)
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    
    return float(loss)
