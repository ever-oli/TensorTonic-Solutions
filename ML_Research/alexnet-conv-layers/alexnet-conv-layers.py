import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # 1. Extract the batch size from the input image
    # Input shape: (Batch, 224, 224, 3)
    batch_size = image.shape[0]
    
    # 2. Define the output dimensions based on the AlexNet architecture
    # Filters: 96
    # Spatial output: 55x55 (calculated via (224 + 2*padding - 11) / 4 + 1)
    # Note: Hint 1 indicates padding=2 is used to reach the 55x55 output.
    output_h = 55
    output_w = 55
    num_filters = 96
    
    # 3. Create a representative output tensor of the correct shape
    # Output shape: (Batch, 55, 55, 96)
    output = np.zeros((batch_size, output_h, output_w, num_filters))
    
    return output