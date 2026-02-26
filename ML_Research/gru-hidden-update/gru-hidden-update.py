import numpy as np

def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray,
                  z_t: np.ndarray) -> np.ndarray:
    """Compute final state: h_t = z*h_prev + (1-z)*h_tilde"""
    
    # 1. Compute the contribution of the old state
    # When z_t is 1, we keep the previous state entirely
    keep_old = z_t * h_prev
    
    # 2. Compute the contribution of the new candidate state
    # When z_t is 0, (1 - z_t) is 1, and we use the candidate entirely
    use_new = (1 - z_t) * h_tilde
    
    # 3. Sum the two parts for the final update
    h_t = keep_old + use_new
    
    return h_t