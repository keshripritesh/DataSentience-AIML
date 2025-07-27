from PIL import Image
import numpy as np
import cv2

def preprocess_image(input, target_size=(30, 30)):
    """Load and preprocess image for model prediction.
    Accepts either file path (str) or image array (numpy array)."""
    if isinstance(input, str):
        # Input is a file path
        img = Image.open(input)
    else:
        # Input is a numpy array
        if input.shape[2] == 3:  # Convert BGR to RGB if needed
            img = Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(input)
    
    img = img.convert('RGB')
    img = img.resize(target_size)
    return np.array(img)