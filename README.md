# Image Stitching

A Python implementation of image panorama stitching using Harris corner detection, SIFT features, and homography.

## Overview

This project stitches multiple overlapping images together to create seamless panoramic images. The implementation uses:

- **Harris Corner Detection**: Identifies feature points in images
- **SIFT (Scale-Invariant Feature Transform)**: Extracts robust descriptors at corner locations
- **Feature Matching**: Uses brute-force matcher with Lowe's ratio test
- **Homography Estimation**: Computes transformation matrices using RANSAC
- **Multi-band Blending**: Creates smooth transitions between images

## Project Structure

```
.
├── image_stitching.py              # Refactored Python module with core functionality
├── image_stitching_refactored.ipynb # Clean, documented Jupyter notebook
├── image pairs/                    # Sample image pairs for stitching
├── Image Stitching.ipynb           # Original notebook (legacy)
├── Image Stitching v2.ipynb        # Original v2 notebook (legacy)
└── README.md                       # This file
```

## Requirements

```bash
pip install opencv-contrib-python numpy matplotlib
```

**Note**: `opencv-contrib-python` is required for SIFT features (in `cv2.xfeatures2d`).

## Usage

### Using the Python Module

```python
from image_stitching import ImageStitcher

# Initialize the stitcher
stitcher = ImageStitcher()

# Stitch two images
result = stitcher.stitch_images('image1.jpg', 'image2.jpg')

# Save or display the result
import matplotlib.pyplot as plt
plt.imshow(result)
plt.show()
```

### Using the Jupyter Notebook

1. Open `image_stitching_refactored.ipynb` in Jupyter
2. Run the cells sequentially to:
   - Load and visualize image pairs
   - Detect corners using Harris detector
   - Stitch images together
   - View the panoramic results

### Customizing Parameters

```python
# Create stitcher with custom parameters
stitcher = ImageStitcher(
    harris_threshold=0.01,      # Corner detection sensitivity (0.0-1.0)
    match_ratio=0.85,           # Feature matching ratio threshold
    smoothing_window=188        # Blending window size
)

result = stitcher.stitch_images('left.jpg', 'right.jpg')
```

## API Reference

### `ImageStitcher` Class

Main class for image stitching operations.

#### Methods

- **`__init__(harris_threshold, match_ratio, smoothing_window)`**
  - Initialize stitcher with custom parameters
  
- **`detect_corners(image_path)`**
  - Detect corners using Harris corner detector
  - Returns: `(visualized_image, corner_coordinates)`

- **`extract_sift_features(image_path, corner_locations)`**
  - Extract SIFT descriptors at corner locations
  - Returns: `(keypoints, descriptors)`

- **`compute_homography(descriptors1, keypoints1, descriptors2, keypoints2)`**
  - Compute homography matrix between feature sets
  - Returns: `homography_matrix` (3x3 numpy array)

- **`stitch_images(image_path1, image_path2)`**
  - Complete pipeline to stitch two images
  - Returns: RGB panorama image

## Algorithm Details

### 1. Corner Detection
The Harris corner detector identifies feature points by analyzing the local image gradient structure. Corners are detected where the image intensity changes significantly in multiple directions.

### 2. Feature Description
SIFT (Scale-Invariant Feature Transform) extracts 128-dimensional descriptors at each corner location. These descriptors are invariant to scale, rotation, and illumination changes.

### 3. Feature Matching
- Uses brute-force matcher to find correspondences
- Applies Lowe's ratio test (threshold: 0.85) to filter unreliable matches
- Requires minimum 8 good matches for homography computation

### 4. Homography Estimation
- RANSAC algorithm robustly estimates the transformation matrix
- Outlier threshold: 5.0 pixels
- Handles mismatches and improves accuracy

### 5. Image Blending
- Creates gradient masks for smooth transitions
- Blending window size: 188 pixels (configurable)
- Automatically crops to non-zero regions

## Examples

The `image pairs/` directory contains sample image pairs:
- `image pairs_01_*.jpg` - Sample 1
- `image pairs_02_*.png` - Sample 2
- `image pairs_03_*.jpg` - Sample 3
- `image pairs_04_*.jpg` - Sample 4

## Refactoring Improvements

This refactored version includes several improvements over the original code:

### Code Quality
- ✅ **Modular design**: Core logic extracted into reusable `ImageStitcher` class
- ✅ **Clear naming**: Descriptive function and variable names
- ✅ **Type hints**: Added for better code documentation
- ✅ **Docstrings**: Comprehensive documentation for all public methods
- ✅ **Constants**: Magic numbers replaced with named constants

### Maintainability
- ✅ **Error handling**: Proper validation and error messages
- ✅ **Configurable paths**: No hardcoded file paths
- ✅ **DRY principle**: Eliminated code duplication
- ✅ **Consistent formatting**: PEP 8 style guidelines

### Usability
- ✅ **Simple API**: Easy-to-use `stitch_images()` function
- ✅ **Flexible parameters**: Customizable algorithm parameters
- ✅ **Better visualization**: Cleaner notebook with helper functions
- ✅ **Documentation**: README with usage examples

## Limitations

- Requires OpenCV with contrib modules for SIFT
- Works best with images that have significant overlap (>30%)
- Assumes images are taken from approximately the same viewpoint
- Currently supports only pairwise stitching (not multi-image panoramas)

## Future Enhancements

Potential improvements for future versions:
- Support for multi-image stitching (>2 images)
- Alternative feature detectors (ORB, AKAZE)
- Automatic image alignment detection
- GPU acceleration for faster processing
- Command-line interface
- Web-based interface

## License

This project is for educational purposes. Please ensure you have appropriate licenses for any images you process.

## Credits

Developed as part of a Computer Vision assignment demonstrating image stitching techniques.
