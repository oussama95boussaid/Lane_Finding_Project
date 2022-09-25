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
- <a href= "Camera_Calibration.ipynb">Camera_Calibration.ipynb</a> code for calibration of the camera
- <a href= "Color_transform_&_Gradient_Threshold.ipynb">Color_transform_&_Gradient_Threshold.ipynb</a> code for correction of distortion & calculation of thresholds & Color transform
- <a href= "Perspective_Transformation.ipynb">Perspective_Transformation.ipynb</a> code for perspective transformation
- <a href= "Lane_Finding_Project.ipynb">Lane_Finding_Project.ipynb</a> Code for all the project  
- <a href= "project_video_final.mp4">project video result</a>


# Camera Calibration

Each camera has a certain lens that distorts the captured image in a certain way compared to reality. Because we want to capture the view of the surroundings as accurately as possible, we have to correct this distortion. OpenCV provides some very useful functions to accomplish this task.

The code for this step  in the file called <a href= "Camera_Calibration.ipynb">Camera_Calibration.ipynb</a>.

I start by preparing **"object points"**, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, **objp** is just a replicated array of coordinates, and **objpoints** will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. To do so I use the **cv2.findChessboardCorners()** function provided by OpenCV. **imgpoints** will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
