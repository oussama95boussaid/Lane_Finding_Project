# **Lane Finding Project**

The goal of this project is to create a powerful pipeline to detect lane lines with raw images from a car's dash cam. The pipeline will visually display the lane boundaries, numerically giving a numerical estimate of the lane curvature.

<img src ="output_images/output.png">

# Dependencies 

**To execute the pipeline, the following dependencies are necessary :**

- numpy
- cv2
- glob
- matplotlib
- moviepy.editor

# The Project

**The goals / steps of this project are the following :**

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**Overview of Files :**

My project includes the following files :

- <a href= "README.md">README.md</a> (writeup report) documentation of the results
pipeline.py contains the pipeline for lane line detection
camera_calibration.py code for calibration of the camera
undistorter.py code for correction of distortion
threshold.py code for calculation of thresholds
perspective_trafo.py code for perspective transformation
lanefinder.py code for finding and drawing lane lines
curvatuere.py code for calculation of curvature and the position of the car within the lane lines
image_util.py code for loading and saving images and for calculation of the visalization of original and processed images.
project video result
