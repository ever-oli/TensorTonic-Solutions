import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    """Extract a random crop from the image."""
    h, w, _ = image.shape
    
    # 1. Generate random top/left coordinates (Hint 1)
    # The maximum possible starting index is (image_size - crop_size)
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    
    # 2. Extract the patch using slicing
    return image[top:top+crop_size, left:left+crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Randomly flip image horizontally with probability p."""
    # 3. Determine if the flip should occur based on probability p
    if np.random.random() < p:
        # 4. Perform horizontal flip using slicing (Hint 2)
        # [:, ::-1, :] reverses the elements along the width axis
        return image[:, ::-1, :]
    
    return image