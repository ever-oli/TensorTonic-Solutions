import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # 1. Compute the spectral norm (L2 norm) of the weight matrix
    # Using np.linalg.norm with ord=2 as suggested in Hint 1
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    
    # 2. Initialize the simulation
    # The first norm (at step T) is relative and starts at 1.0
    norms = [1.0]
    current_norm = 1.0
    
    # 3. Simulate gradient magnitude over T-1 additional time steps
    # Each step backwards involves another multiplication by the weights
    for _ in range(T - 1):
        current_norm *= spectral_norm
        norms.append(current_norm)
        
    return norms