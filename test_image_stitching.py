"""
Unit tests for the image_stitching module.

Note: These tests require opencv-contrib-python to be installed.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestImageStitcher(unittest.TestCase):
    """Test cases for the ImageStitcher class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock cv2 to avoid import errors during testing without opencv
        self.cv2_mock = Mock()
        sys.modules['cv2'] = self.cv2_mock
        sys.modules['cv2.xfeatures2d'] = Mock()
        
        # Now we can import
        from image_stitching import ImageStitcher
        self.stitcher = ImageStitcher()

    def tearDown(self):
        """Clean up after tests."""
        # Remove mocked modules
        if 'cv2' in sys.modules:
            del sys.modules['cv2']
        if 'cv2.xfeatures2d' in sys.modules:
            del sys.modules['cv2.xfeatures2d']
        if 'image_stitching' in sys.modules:
            del sys.modules['image_stitching']

    def test_initialization(self):
        """Test ImageStitcher initialization with default parameters."""
        from image_stitching import ImageStitcher
        stitcher = ImageStitcher()
        self.assertEqual(stitcher.harris_threshold, 0.01)
        self.assertEqual(stitcher.match_ratio, 0.85)
        self.assertEqual(stitcher.smoothing_window, 188)

    def test_initialization_custom_params(self):
        """Test ImageStitcher initialization with custom parameters."""
        from image_stitching import ImageStitcher
        stitcher = ImageStitcher(
            harris_threshold=0.02,
            match_ratio=0.75,
            smoothing_window=200
        )
        self.assertEqual(stitcher.harris_threshold, 0.02)
        self.assertEqual(stitcher.match_ratio, 0.75)
        self.assertEqual(stitcher.smoothing_window, 200)

    def test_create_blend_mask_left(self):
        """Test blend mask creation for left image."""
        from image_stitching import ImageStitcher
        stitcher = ImageStitcher(smoothing_window=100)
        
        img1 = np.zeros((100, 200, 3))
        img2 = np.zeros((100, 150, 3))
        
        mask = stitcher.create_blend_mask(img1, img2, 'left_image')
        
        # Check mask shape
        expected_height = 100
        expected_width = 200 + 150
        self.assertEqual(mask.shape, (expected_height, expected_width, 3))
        
        # Check that mask has 3 channels
        self.assertEqual(mask.shape[2], 3)

    def test_create_blend_mask_right(self):
        """Test blend mask creation for right image."""
        from image_stitching import ImageStitcher
        stitcher = ImageStitcher(smoothing_window=100)
        
        img1 = np.zeros((100, 200, 3))
        img2 = np.zeros((100, 150, 3))
        
        mask = stitcher.create_blend_mask(img1, img2, 'right_image')
        
        # Check mask shape
        expected_height = 100
        expected_width = 200 + 150
        self.assertEqual(mask.shape, (expected_height, expected_width, 3))

    def test_create_blend_mask_invalid_type(self):
        """Test blend mask creation with invalid type."""
        from image_stitching import ImageStitcher
        stitcher = ImageStitcher()
        
        img1 = np.zeros((100, 200, 3))
        img2 = np.zeros((100, 150, 3))
        
        with self.assertRaises(ValueError):
            stitcher.create_blend_mask(img1, img2, 'invalid_type')

    def test_constants_defined(self):
        """Test that all required constants are defined."""
        from image_stitching import (
            HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K, HARRIS_THRESHOLD,
            SIFT_KEYPOINT_SIZE, MATCH_RATIO_THRESHOLD, MIN_GOOD_MATCHES,
            RANSAC_THRESHOLD, SMOOTHING_WINDOW_SIZE
        )
        
        # Verify constants have reasonable values
        self.assertEqual(HARRIS_BLOCK_SIZE, 2)
        self.assertEqual(HARRIS_KSIZE, 3)
        self.assertAlmostEqual(HARRIS_K, 0.04)
        self.assertAlmostEqual(HARRIS_THRESHOLD, 0.01)
        self.assertEqual(SIFT_KEYPOINT_SIZE, 10)
        self.assertAlmostEqual(MATCH_RATIO_THRESHOLD, 0.85)
        self.assertEqual(MIN_GOOD_MATCHES, 8)
        self.assertAlmostEqual(RANSAC_THRESHOLD, 5.0)
        self.assertEqual(SMOOTHING_WINDOW_SIZE, 188)


class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock cv2
        self.cv2_mock = Mock()
        sys.modules['cv2'] = self.cv2_mock
        sys.modules['cv2.xfeatures2d'] = Mock()

    def tearDown(self):
        """Clean up after tests."""
        if 'cv2' in sys.modules:
            del sys.modules['cv2']
        if 'cv2.xfeatures2d' in sys.modules:
            del sys.modules['cv2.xfeatures2d']
        if 'image_stitching' in sys.modules:
            del sys.modules['image_stitching']

    def test_stitch_images_function_exists(self):
        """Test that the convenience function exists."""
        from image_stitching import stitch_images
        self.assertTrue(callable(stitch_images))


if __name__ == '__main__':
    unittest.main()
