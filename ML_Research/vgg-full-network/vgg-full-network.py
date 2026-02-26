import numpy as np

# Helper functions are provided by the platform:
# - vgg_features(x, config)
# - vgg_classifier(features, num_classes)

def vgg16(x: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    """
    Implement the complete VGG-16 network.
    """
    # 1. Define the standard VGG-16 configuration (Config D)
    # 13 convolutional layers + 5 max-pooling layers
    vgg16_config = [
        64, 64, 'M',           # Block 1
        128, 128, 'M',         # Block 2
        256, 256, 256, 'M',    # Block 3
        512, 512, 512, 'M',    # Block 4
        512, 512, 512, 'M'     # Block 5
    ]
    
    # 2. Extract features using the convolutional backbone
    # Input (B, 224, 224, 3) -> Output (B, 7, 7, 512)
    features = vgg_features(x, vgg16_config)
    
    # 3. Pass features through the classifier head
    # The helper handles flattening and the three dense layers
    # Output (B, num_classes)
    logits = vgg_classifier(features, num_classes)
    
    return logits