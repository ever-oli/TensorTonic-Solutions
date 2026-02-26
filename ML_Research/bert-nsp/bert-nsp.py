import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples with a 50/50 split of IsNext and NotNext pairs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    examples = []
    
    while len(examples) < num_examples:
        # Select a random document
        doc_idx = random.randint(0, len(documents) - 1)
        document = documents[doc_idx]
        
        # Ensure the document has at least two sentences for a potential positive pair
        if len(document) < 2:
            continue
            
        # Select a random sentence index (not the last one)
        sent_idx = random.randint(0, len(document) - 2)
        
        # 50% chance for IsNext (Positive Pair)
        if random.random() < 0.5:
            examples.append((document[sent_idx], document[sent_idx + 1], 1))
        
        # 50% chance for NotNext (Negative Pair)
        else:
            # Pick a random sentence from a different document if possible
            if len(documents) > 1:
                random_doc_idx = doc_idx
                while random_doc_idx == doc_idx:
                    random_doc_idx = random.randint(0, len(documents) - 1)
                random_document = documents[random_doc_idx]
            else:
                random_document = document
                
            random_sent_idx = random.randint(0, len(random_document) - 1)
            examples.append((document[sent_idx], random_document[random_sent_idx], 0))
            
    return examples[:num_examples]

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        # Weight matrix for 2 classes: IsNext (1) and NotNext (0)
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext probability using the [CLS] token's hidden state.
        cls_hidden shape: (batch_size, hidden_size)
        """
        # Linear transformation: logits = xW + b
        logits = np.dot(cls_hidden, self.W) + self.b
        return logits

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)