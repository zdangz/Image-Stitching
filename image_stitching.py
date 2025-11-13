"""
Image Stitching Module

This module provides functionality for stitching multiple images together to create
panoramic images using Harris corner detection, SIFT feature extraction, and homography.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List


# Constants
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01

SIFT_KEYPOINT_SIZE = 10

MATCH_RATIO_THRESHOLD = 0.85
MIN_GOOD_MATCHES = 8
RANSAC_THRESHOLD = 5.0

SMOOTHING_WINDOW_SIZE = 188


class ImageStitcher:
    """
    A class for stitching images together using feature matching and homography.
    """

    def __init__(self, 
                 harris_threshold: float = HARRIS_THRESHOLD,
                 match_ratio: float = MATCH_RATIO_THRESHOLD,
                 smoothing_window: int = SMOOTHING_WINDOW_SIZE):
        """
        Initialize the ImageStitcher with configurable parameters.

        Args:
            harris_threshold: Threshold for Harris corner detection (0.0 to 1.0)
            match_ratio: Ratio threshold for feature matching (0.0 to 1.0)
            smoothing_window: Window size for blending smoothing
        """
        self.harris_threshold = harris_threshold
        self.match_ratio = match_ratio
        self.smoothing_window = smoothing_window

    def detect_corners(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect corners in an image using Harris corner detector.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (visualized_image, corner_coordinates)
                - visualized_image: RGB image with corners marked in red
                - corner_coordinates: Array of (row, col) coordinates of detected corners
        """
        # Read and convert image to grayscale
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # Detect corners using Harris detector
        harris_response = cv2.cornerHarris(
            gray, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K
        )

        # Get coordinates of detected corners
        threshold_value = self.harris_threshold * harris_response.max()
        corner_coordinates = np.argwhere(harris_response > threshold_value)

        # Dilate for visualization (not used for actual detection)
        harris_response = cv2.dilate(harris_response, None)

        # Mark corners in red
        image[harris_response > threshold_value] = [0, 0, 255]

        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb, corner_coordinates

    def extract_sift_features(self, image_path: str, 
                             corner_locations: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT descriptors at specified corner locations.

        Args:
            image_path: Path to the input image
            corner_locations: Array of (row, col) corner coordinates

        Returns:
            Tuple of (keypoints, descriptors)
                - keypoints: List of cv2.KeyPoint objects
                - descriptors: Array of SIFT descriptors
        """
        sift = cv2.xfeatures2d.SIFT_create()
        
        # Convert corner locations to keypoints
        keypoints = []
        for corner in corner_locations:
            # Note: KeyPoint takes (x, y) which is (col, row)
            keypoint = cv2.KeyPoint(
                float(corner[1]),  # x = column
                float(corner[0]),  # y = row
                SIFT_KEYPOINT_SIZE
            )
            keypoints.append(keypoint)

        # Read image in grayscale for SIFT
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Compute descriptors
        keypoints, descriptors = sift.compute(image, keypoints)

        return keypoints, descriptors

    def compute_homography(self, descriptors1: np.ndarray, keypoints1: List,
                          descriptors2: np.ndarray, keypoints2: List) -> np.ndarray:
        """
        Compute homography matrix between two sets of features.

        Args:
            descriptors1: SIFT descriptors from first image
            keypoints1: Keypoints from first image
            descriptors2: SIFT descriptors from second image
            keypoints2: Keypoints from second image

        Returns:
            Homography matrix (3x3 numpy array)
        """
        # Match features using brute force matcher
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to filter good matches
        good_points = []
        for match1, match2 in raw_matches:
            if match1.distance < self.match_ratio * match2.distance:
                good_points.append((match1.trainIdx, match1.queryIdx))

        # Ensure we have enough matches
        if len(good_points) < MIN_GOOD_MATCHES:
            raise ValueError(
                f"Not enough good matches found: {len(good_points)} "
                f"(minimum required: {MIN_GOOD_MATCHES})"
            )

        # Extract matched keypoint coordinates
        points_img1 = np.float32(
            [keypoints1[query_idx].pt for (_, query_idx) in good_points]
        )
        points_img2 = np.float32(
            [keypoints2[train_idx].pt for (train_idx, _) in good_points]
        )

        # Compute homography using RANSAC
        homography_matrix, _ = cv2.findHomography(
            points_img2, points_img1, cv2.RANSAC, RANSAC_THRESHOLD
        )

        return homography_matrix

    def create_blend_mask(self, img1: np.ndarray, img2: np.ndarray, 
                         mask_type: str) -> np.ndarray:
        """
        Create a blending mask for smooth image transitions.

        Args:
            img1: First image (left)
            img2: Second image (right)
            mask_type: Either 'left_image' or 'right_image'

        Returns:
            3-channel blending mask
        """
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]

        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        offset = int(self.smoothing_window / 2)
        barrier = width_img1 - offset

        mask = np.zeros((height_panorama, width_panorama))

        if mask_type == 'left_image':
            # Gradual transition from 1 to 0
            mask[:, barrier - offset:barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, :barrier - offset] = 1
        elif mask_type == 'right_image':
            # Gradual transition from 0 to 1
            mask[:, barrier - offset:barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset:] = 1
        else:
            raise ValueError(f"Invalid mask_type: {mask_type}")

        return cv2.merge([mask, mask, mask])

    def blend_images(self, img1: np.ndarray, img2: np.ndarray, 
                    homography_matrix: np.ndarray) -> np.ndarray:
        """
        Blend two images using the homography matrix and gradient masks.

        Args:
            img1: First image (left)
            img2: Second image (right)
            homography_matrix: Transformation matrix for img2

        Returns:
            Blended panorama image
        """
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]

        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # Create panoramas for both images
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_blend_mask(img1, img2, mask_type='left_image')
        panorama1[0:height_img1, 0:width_img1, :] = img1
        panorama1 *= mask1

        # Warp and blend second image
        mask2 = self.create_blend_mask(img1, img2, mask_type='right_image')
        panorama2 = cv2.warpPerspective(
            img2, homography_matrix, (width_panorama, height_panorama)
        ) * mask2

        # Combine panoramas
        result = panorama1 + panorama2

        # Crop to non-zero region
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result

    def stitch_images(self, image_path1: str, image_path2: str) -> np.ndarray:
        """
        Stitch two images together to create a panorama.

        Args:
            image_path1: Path to the first (left) image
            image_path2: Path to the second (right) image

        Returns:
            Stitched panorama image in RGB format
        """
        # Detect corners
        _, corner_locations1 = self.detect_corners(image_path1)
        _, corner_locations2 = self.detect_corners(image_path2)

        # Extract SIFT features
        keypoints1, descriptors1 = self.extract_sift_features(
            image_path1, corner_locations1
        )
        keypoints2, descriptors2 = self.extract_sift_features(
            image_path2, corner_locations2
        )

        # Compute homography
        homography_matrix = self.compute_homography(
            descriptors1, keypoints1, descriptors2, keypoints2
        )

        # Load images
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")

        # Blend images
        result = self.blend_images(img1, img2, homography_matrix)
        result = result.astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result


def stitch_images(image_path1: str, image_path2: str, **kwargs) -> np.ndarray:
    """
    Convenience function to stitch two images together.

    Args:
        image_path1: Path to the first (left) image
        image_path2: Path to the second (right) image
        **kwargs: Additional parameters for ImageStitcher

    Returns:
        Stitched panorama image in RGB format
    """
    stitcher = ImageStitcher(**kwargs)
    return stitcher.stitch_images(image_path1, image_path2)
