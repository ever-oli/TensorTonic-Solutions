import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply BERT's MLM masking strategy.
    """
    if seed is not None:
        np.random.seed(seed)
    
    masked_ids = token_ids.copy()
    # Initialize labels with -100 (ignored in loss calculation)
    labels = np.full(token_ids.shape, -100)
    
    # 1. Identify valid candidates for masking (exclude [CLS], [SEP], [PAD])
    # Assuming standard BERT: 101: [CLS], 102: [SEP], 0: [PAD]
    mask_eligible = ~np.isin(token_ids, [101, 102, 0])
    
    # 2. Randomly select 15% of eligible tokens
    probability_matrix = np.random.rand(*token_ids.shape)
    mask_indices = (probability_matrix < mask_prob) & mask_eligible
    
    # Set labels for the positions we are predicting
    labels[mask_indices] = token_ids[mask_indices]
    
    # 3. Apply the 80-10-10 strategy to selected indices
    random_dispatch = np.random.rand(*token_ids.shape)
    
    # 80% of the time, replace with [MASK]
    indices_replaced = mask_indices & (random_dispatch < 0.8)
    masked_ids[indices_replaced] = mask_token_id
    
    # 10% of the time, replace with random token
    indices_random = mask_indices & (random_dispatch >= 0.8) & (random_dispatch < 0.9)
    masked_ids[indices_random] = np.random.randint(0, vocab_size, size=np.sum(indices_random))
    
    # The remaining 10% (random_dispatch >= 0.9) are left unchanged in masked_ids
    
    return masked_ids, labels, mask_indices

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token probabilities across the vocabulary.
        hidden_states shape: (batch, seq_len, hidden_size)
        """
        # Linear transformation: logits = xW + b
        logits = np.dot(hidden_states, self.W) + self.b
        return logits