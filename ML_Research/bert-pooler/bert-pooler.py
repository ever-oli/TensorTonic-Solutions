import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Pool the [CLS] token representation.
        hidden_states shape: (batch, seq_len, hidden_size)
        """
        # 1. Extract the [CLS] hidden state at position 0
        # shape: (batch, hidden_size)
        cls_token_tensor = hidden_states[:, 0]
        
        # 2. Apply linear transformation: h_pooled = W * h_CLS + b
        pooled_output = np.dot(cls_token_tensor, self.W) + self.b
        
        # 3. Apply tanh activation to bound values between [-1, 1]
        return tanh(pooled_output)

class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        # Classifier weights shape: (hidden_size, num_classes)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
        self.bias = np.zeros(num_classes)
    
    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Classify sequences by pooling and applying a linear layer.
        """
        # 1. Get the pooled representation from the [CLS] token
        # shape: (batch, hidden_size)
        pooled_output = self.pooler.forward(hidden_states)
        
        # 2. Apply dropout during training (simple mask implementation)
        if training:
            mask = (np.random.rand(*pooled_output.shape) > self.dropout_prob)
            pooled_output = (pooled_output * mask) / (1.0 - self.dropout_prob)
            
        # 3. Project to number of classes (logits)
        # shape: (batch, num_classes)
        logits = np.dot(pooled_output, self.classifier) + self.bias
        
        return logits
