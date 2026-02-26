import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """Update cell state: C_t = f_t * C_prev + i_t * c_tilde"""
    
    # 1. Element-wise multiply forget gate with previous cell state
    # This determines what part of the old memory to keep
    forget_part = f_t * C_prev
    
    # 2. Element-wise multiply input gate with candidate memory
    # This determines what new information to add
    input_part = i_t * c_tilde
    
    # 3. Add the two results to get the new cell state
    C_t = forget_part + input_part
    
    return C_t