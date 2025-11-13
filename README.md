# Image Stitching

A Python implementation of automatic image panorama stitching using computer vision techniques. This project combines multiple overlapping images into a seamless panoramic view using Harris corner detection, SIFT descriptors, and homography transformation.

## Features

- **Harris Corner Detection**: Identifies key feature points in images for matching
- **SIFT Descriptors**: Generates robust feature descriptors invariant to scale and rotation
- **Homography Estimation**: Calculates geometric transformations between image pairs using RANSAC
- **Multi-band Blending**: Seamlessly blends overlapping regions to create natural-looking panoramas
- **Automatic Alignment**: Automatically aligns and stitches image pairs with overlapping content

## Algorithm Overview

The image stitching process follows these steps:

1. **Feature Detection**: Uses Harris corner detection to identify keypoints in both images
2. **Feature Description**: Computes SIFT descriptors for detected keypoints
3. **Feature Matching**: Matches descriptors between images using Brute Force matcher with k-nearest neighbors
4. **Homography Calculation**: Estimates the geometric transformation using RANSAC to filter outliers
5. **Image Warping**: Transforms the second image to align with the first using the homography matrix
6. **Blending**: Combines images with smooth transitions using gradient-based alpha blending

## Requirements

- Python 3.x
- OpenCV with contrib modules (for SIFT)
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zdangz/Image-Stitching.git
cd Image-Stitching
```

2. Install required dependencies:
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

**Note**: SIFT requires `opencv-contrib-python` as it's a patented algorithm not included in the standard OpenCV distribution.

## Usage

### Running the Jupyter Notebooks

The project includes several Jupyter notebooks demonstrating the image stitching pipeline:

1. Open the notebook:
```bash
jupyter notebook "Image Stitching v2.ipynb"
```

2. Run all cells to see the complete stitching process

### Using the Core Functions

```python
import cv2
import numpy as np

# Load your image pair
img1_path = "image pairs/image pairs_01_01.jpg"
img2_path = "image pairs/image pairs_01_02.jpg"

# Stitch images
result = image_stitch(img1_path, img2_path)

# Display result
import matplotlib.pyplot as plt
plt.imshow(result)
plt.axis('off')
plt.show()

# Save result
cv2.imwrite('panorama_result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
```

### Function Reference

#### `harris(img_dir)`
Detects corner points in an image using Harris corner detector.
- **Input**: Path to image file
- **Output**: Tuple of (annotated image, corner coordinates)

#### `sift(imgdir, corner_loc)`
Computes SIFT descriptors for given corner locations.
- **Input**: Image path and corner coordinates
- **Output**: Tuple of (keypoints, descriptors)

#### `homography(descriptor1, keypoints1, descriptor2, keypoints2)`
Calculates homography matrix between two sets of matched features.
- **Input**: Descriptors and keypoints from both images
- **Output**: 3x3 homography transformation matrix

#### `blending(img1, img2, H)`
Blends two images using the homography matrix with gradient-based alpha blending.
- **Input**: Two images and homography matrix
- **Output**: Stitched panorama image

#### `image_stitch(imgdir1, imgdir2)`
Complete pipeline function that stitches two images.
- **Input**: Paths to two image files
- **Output**: Stitched panoramic image

## Sample Image Pairs

The repository includes four sample image pairs in the `image pairs/` directory:

- **Pair 1** (`image pairs_01_01.jpg`, `image pairs_01_02.jpg`): Outdoor landscape scene
- **Pair 2** (`image pairs_02_01.png`, `image pairs_02_02.png`): Architectural view
- **Pair 3** (`image pairs_03_01.jpg`, `image pairs_03_02.jpg`): Building exterior
- **Pair 4** (`image pairs_04_01.jpg`, `image pairs_04_02.jpg`): Natural scenery

All sample pairs have sufficient overlap (typically 20-40%) for successful stitching.

## Project Structure

```
Image-Stitching/
├── Image Stitching v2.ipynb          # Main implementation notebook (recommended)
├── Image Stitching.ipynb             # Earlier version
├── image_stitching-submission.ipynb  # Submission version
├── image_stitching_draft_git.ipynb   # Development draft
├── image pairs/                      # Sample image pairs
│   ├── image pairs_01_01.jpg
│   ├── image pairs_01_02.jpg
│   ├── image pairs_02_01.png
│   ├── image pairs_02_02.png
│   ├── image pairs_03_01.jpg
│   ├── image pairs_03_02.jpg
│   ├── image pairs_04_01.jpg
│   └── image pairs_04_02.jpg
├── README.md                         # This file
└── DEVELOPER_GUIDE.md                # Detailed development documentation
```

## Technical Details

### Harris Corner Detection
- Block size: 2
- Sobel kernel size: 3
- Harris parameter k: 0.04
- Corner threshold: 1% of maximum corner response

### SIFT Descriptor
- Keypoint size: 10 pixels
- Uses corner locations from Harris detection as SIFT keypoints

### Feature Matching
- Algorithm: Brute Force matcher with k-NN (k=2)
- Lowe's ratio test: 0.85 (filters ambiguous matches)
- Minimum matches required: 8

### Homography Estimation
- Method: RANSAC (Random Sample Consensus)
- RANSAC threshold: 5 pixels
- Minimum inliers: Automatically determined by OpenCV

### Blending
- Smoothing window size: 188 pixels
- Method: Linear gradient alpha blending in the overlap region
- Prevents visible seams between stitched images

## Limitations and Considerations

- Images must have sufficient overlap (recommended: 20-40%)
- Works best with images taken from the same viewpoint (pure rotation)
- Images should be captured with similar exposure and lighting
- SIFT is patented in some countries; consider alternatives like ORB for commercial use
- Performance depends on the number and quality of detected features

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is available for educational and research purposes.

## Acknowledgments

- OpenCV library for computer vision algorithms
- Harris corner detection algorithm by Chris Harris and Mike Stephens
- SIFT algorithm by David Lowe
- Computer Vision course assignment project

## References

- Harris, C., & Stephens, M. (1988). "A Combined Corner and Edge Detector"
- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- Szeliski, R. (2010). "Computer Vision: Algorithms and Applications"

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.
