import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """
    Apply dropout to input.
    x: input tensor
    p: probability of dropping a neuron
    training: boolean flag to indicate training or inference mode
    """
    # 1. Return input unchanged during evaluation/inference
    if not training or p == 0:
        return x
    
    # 2. Create binary mask (m) where 1 appears with probability 1 - p
    # Use np.random.binomial as suggested in Hint 1
    mask = np.random.binomial(1, 1 - p, size=x.shape)
    
    # 3. Apply mask and scale by 1 / (1 - p)
    # Scaling ensures the expected magnitude of the signal is preserved (Hint 2)
    return (x * mask) / (1 - p)