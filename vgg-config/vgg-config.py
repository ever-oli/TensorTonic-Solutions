import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant.
    The list contains integers for conv out_channels and 'M' for MaxPool.
    """
    # Define configurations for each variant
    configs = {
        "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    
    # Normalize input to lowercase to handle case-insensitivity
    key = variant.lower()
    
    return configs.get(key, [])