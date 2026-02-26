import numpy as np
from typing import List, Optional

class MockBertEncoder:
    """Simulated BERT encoder with 12 layers."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Each layer just adds a small transformation
        self.layers = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers
    
    def freeze_layers(self, layer_indices: List[int]):
        """Freeze specified layers (no gradient updates)."""
        for idx in layer_indices:
            if 0 <= idx < self.num_layers:
                self.layer_frozen[idx] = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        self.layer_frozen = [False] * self.num_layers
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        x = embeddings
        for i, layer in enumerate(self.layers):
            # In this mock, we compute the transformation for all layers.
            # In a real framework, frozen layers would skip gradient computation.
            x = x @ layer + x  # Simplified residual
        return x

class BertForSequenceClassification:
    """BERT with sequence-level classification head (e.g. Sentiment)."""
    
    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.bias = np.zeros(num_labels)
        self.freeze_bert = freeze_bert
        
        if freeze_bert:
            # Freeze all 12 layers of the encoder
            self.encoder.freeze_layers(list(range(12)))
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for sequence classification using the [CLS] token.
        """
        # 1. Get hidden states from encoder: (batch, seq_len, hidden_size)
        hidden_states = self.encoder.forward(embeddings)
        
        # 2. Extract [CLS] representation at position 0: (batch, hidden_size)
        cls_representation = hidden_states[:, 0, :]
        
        # 3. Linear transformation: y = W * h_cls + b
        logits = cls_representation @ self.classifier + self.bias
        return logits

class BertForTokenClassification:
    """BERT with token-level classification (e.g. NER, POS tagging)."""
    
    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.bias = np.zeros(num_labels)
    
    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass for token classification applied to all positions.
        """
        # 1. Get hidden states from encoder: (batch, seq_len, hidden_size)
        hidden_states = self.encoder.forward(embeddings)
        
        # 2. Apply classifier to every token position: (batch, seq_len, num_labels)
        logits = hidden_states @ self.classifier + self.bias
        return logits