import numpy as np

def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement VGG's fully connected classifier.
    Input features shape: (Batch, 7, 7, 512)
    Output shape: (Batch, num_classes)
    """
    # 1. Flatten the input
    # Reshape from (B, 7, 7, 512) to (B, 25088)
    batch_size = features.shape[0]
    x = features.reshape(batch_size, -1)
    
    # Helper to initialize weights and perform a Dense + ReLU layer
    def dense_relu(input_data, out_dim):
        in_dim = input_data.shape[1]
        # He initialization for ReLU
        limit = np.sqrt(2 / in_dim)
        w = np.random.randn(in_dim, out_dim) * limit
        b = np.zeros(out_dim)
        # Linear transform + ReLU
        return np.maximum(0, input_data @ w + b)

    # 2. First Hidden Layer: FC1 (4096 units) + ReLU
    x = dense_relu(x, 4096)
    
    # 3. Second Hidden Layer: FC2 (4096 units) + ReLU
    x = dense_relu(x, 4096)
    
    # 4. Final Classification Layer: FC3 (num_classes)
    # No ReLU after the final layer (logits)
    in_dim_final = x.shape[1]
    w_final = np.random.randn(in_dim_final, num_classes) * np.sqrt(2 / in_dim_final)
    b_final = np.zeros(num_classes)
    
    logits = x @ w_final + b_final
    
    return logits