import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Compute BERT embeddings by summing token, position, and segment embeddings.
        """
        # 1. Look up token embeddings
        # shape: (batch_size, seq_len, hidden_size)
        tok_emb = self.token_embeddings[token_ids]
        
        # 2. Create position indices and look up position embeddings
        # positions = [0, 1, 2, ..., seq_len-1]
        seq_len = token_ids.shape[1]
        positions = np.arange(seq_len)
        # shape: (seq_len, hidden_size)
        pos_emb = self.position_embeddings[positions]
        
        # 3. Look up segment embeddings
        # shape: (batch_size, seq_len, hidden_size)
        seg_emb = self.segment_embeddings[segment_ids]
        
        # Sum the components: E = E_token + E_position + E_segment
        # Note: pos_emb (seq_len, hidden_size) will broadcast across the batch dimension
        embeddings = tok_emb + pos_emb + seg_emb
        
        return embeddings
