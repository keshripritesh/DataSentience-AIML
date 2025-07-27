import unittest
from utils import preprocess_image
import numpy as np
from PIL import Image

class TestUtils(unittest.TestCase):
    def test_preprocess_image_file(self):
        img = preprocess_image('test_image.jpg')
        self.assertEqual(img.shape, (30, 30, 3))
        self.assertEqual(img.dtype, np.uint8)
        
    def test_preprocess_image_array(self):
        # Create test image array
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img = preprocess_image(test_array)
        self.assertEqual(img.shape, (30, 30, 3))
        self.assertEqual(img.dtype, np.uint8)