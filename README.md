# Panorama_ImageStitching

## 🎯 Motivation
Panorama image stitching is a fascinating application of computer vision, widely used in photography, virtual tours, and mapping services. Creating seamless panoramic images from multiple overlapping images requires advanced techniques to detect, match, and blend features effectively. This project was inspired by the challenge of comparing different stitching methodologies to achieve optimal results across diverse image sets.

## 📖 About the Project
This project implements a comprehensive pipeline for panorama image stitching using modern computer vision techniques. It focuses on feature detection, matching, transformation estimation, and image blending while evaluating two different blending approaches.

### Key Steps in the Pipeline:
1. **Feature Detection and Description**:  
   - Use **SIFT (Scale-Invariant Feature Transform)** to detect keypoints and extract descriptors from input images.

2. **Feature Matching**:  
   - Employ **FLANN-based KNN matcher** to find strong correspondences between keypoints in overlapping images.

3. **Homography Estimation**:  
   - Apply **RANSAC (Random Sample Consensus)** and **Direct Linear Transformation (DLT)** to compute the homography matrix, ensuring robust transformation by filtering outliers.

4. **Image Stitching**:  
   - Implement and compare two blending methods:
     - **Max Pixel Intensity**: Selects the maximum pixel value for overlapping regions.
     - **Weighted Distance Map**: Applies a weighted blending approach, considering the proximity of pixels to the center of each image.

5. **Evaluation**:  
   - Test the stitching pipeline on various datasets and compare the visual outcomes of the two blending approaches.
  
## ⚙️ Features
- Robust and accurate feature matching using FLANN and SIFT.
- Outlier rejection and accurate homography calculation via RANSAC.
- Two blending approaches with comparative analysis.
- Flexibility to handle diverse datasets with varying levels of overlap and complexity.

  
## 🛠️ Challenges Faced
- **Feature Detection Sensitivity**: Fine-tuning SIFT to ensure robust detection in images with low contrast or significant variations in lighting.
- **Outlier Removal**: RANSAC parameters required careful adjustment to remove noise without discarding true matches.
- **Blending Quality**: Achieving seamless blending in the overlapping region posed challenges due to variations in intensity and color across images.
- **Computational Efficiency**: Processing large image sets with high resolution increased the computational load, necessitating optimization strategies.


## 🚀 Future Work
- **Real-Time Stitching**: Enhance the pipeline to support real-time stitching for live video feeds.
- **Automatic Parameter Tuning**: Develop a mechanism to automatically adjust parameters like RANSAC thresholds based on image characteristics.
- **Machine Learning Integration**: Explore using deep learning models for feature detection and matching to improve robustness.
- **Dynamic Blending**: Implement advanced blending techniques, such as multi-band blending, to handle images with complex lighting conditions better.
- **3D Stitching**: Extend the project to include stitching for spherical panoramas or 3D point clouds.




## Intstruction to run the code:
1. Add the path of images to be stitched in the "image_list[]" in application.py (Line 169). 
    Some sample images that can be stitched are given in "test_images" directory.
2. You can adjust the threshold for Lowe's Ratio (Line 188) and number of RANSAC iterations (Line 189) in application.py.
3. Currently the code blends images using "Max Pixel Intensity" method. This might take some time to run.
4. To use "Weighted Distance Maps" to blend images comment lines 78-95 and uncomment lines 98-145.
5. Run application.py
6. You can observe detected keypoints in "image1_keypoints.jpg" and "image2_keypoints.jpg", and the matched keypoints in "matched_keypoints.jpg".
7. Final stitched panorama is stored in "stitched_panorama.png"
