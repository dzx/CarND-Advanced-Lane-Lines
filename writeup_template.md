# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Calibration"
[image2]: ./output_images/perspective.png "Perspective Measurement"
[image3]: ./output_images/perspective_unwrap.png "Warp Example"
[image4]: ./output_images/gradient.png "Gradient Combination"
[image5]: ./output_images/diags1.png "First Time Fit Visual"
[image6]: ./output_images/diags2.png "Fit Visual"
[image7]: ./output_images/pipeline_test.png "Output"
[image8]: ./output_images/undist_example.png "Undistort Example"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 4 code cells of [this](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Camera-Calibration) IPython notebook section.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image using `cv2.findChessboardCorners()`. Detection is applied on number of chessboard images taken from different angles and converted to gray scale.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to all of the the calibration images using the `cv2.undistort()` function and obtained this results such as this: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like i did here:
![alt text][image8]

I started with test image below and applied my convenience function `undistort()` which is defined 3 cells below [this](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Camera-Calibration) point.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps and functions defined in 2 cells following [this](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Color-and-gradient-transformations) point ).  Here's an example of my output for this step.  

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in 4 cells following [this](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Perspective-Transformation) point. I have identified 4 points on image below as `[[248, 690], [1058, 690], [573, 466], [711, 466]]`

![alt text][image2]

Then I defined 4 corresponding points as `np.float32([[411, img.shape[0]-10], [884, img.shape[0]-10], [411, 160], [884, 160]])` and got warp and unwarp matrices from those two point arrays using `cv2.getPerspectiveTransform()` I tested the result with `cv2.warpPerspective()` function and got this image
![alt text][image3]

Finally I created a convenience function called `warp_image()`. It works in conjunction with warp and unwarp matrices which are left as global variables:

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

It all happens in function `find_lanes()` in [Flatland controller](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Flatland-controller) section of my Jupyter notebook. This lengthy function relies on `find_lane_start()` and `find_line_next()` functions defined [here](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Support-functions-for-lane-detection) to identify pixels that belong to lane lines.
The rest of the code uses functions defined [here](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Lane-detection-validation-functions) to validate polynomial fits and re-evaluate using more reliable method if necessary of discard result altogether. Image below shows diagnostic output in case when prior position of lane lines is not known.

![alt text][image5]

Then this is example of fitting line lanes based on position from prior frame

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 100 through 116 of `find_lanes()` function mentioned above. First identified curves are converted from pixel to meter units, and result is fitted to second degree polynomial. Radus is then calculated like this:
`left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])`

Same is done for right curve, and then result is smoothed by adding 0.25 of it to 0.75 of it's prior value

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Polygon to draw over lane is created in lines 87 through 97 in aforementioned `find_lanes()`. Then it gets perspective-unwarped and blended with original image in pipeline function `process_image()` shown [here](https://render.githubusercontent.com/view/ipynb?commit=095fd42794258e2dc39b03cf9d11c76567625db8&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f647a782f4361724e442d416476616e6365642d4c616e652d4c696e65732f303935666434323739343235386532646333396230336366396431316337363536373632356462382f50344164764c616e654c696e65732e6970796e62&nwo=dzx%2FCarND-Advanced-Lane-Lines&path=P4AdvLaneLines.ipynb&repository_id=92813945&repository_type=Repository#Image-Processing-Pipeline)  
Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/test.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The single bigest problem in this project is that we are using Sobel gradients to infer position of lane lines on the image, and this method is prone to changes in lighting as it heavily relies on getting the appropriate threshold values, and these fluctuate dependind on lighting conditions. I found that saturation component in HLS color space is most useful single component out of all image components I tried and have used disjunction of horizontal, vertical and magnitude gradients, all in conjunction with directional gradient that loosely matches expected direction of lane lines when the road is straight. Conjunction with particular directional gradient can fail in sharp turns when direction of lane lines gets outside of expected range. However, even on fairly straight highway, choosing the optimal component threshold ranges remains a problem as we need to detect as many pixels as possible on dashed lane line, while tuning out pixels from shadows, other cars, seams in the road surface and such. Detecting too few pixels on dashed line means that every outlier has more impact on estimated angle and curvature of the line, and this is why I ecountered much higher magnitude of fluctiations in estimated parameters of dashed line than of the solid line.
Therefore I had to employ logic to discard line parameter estimates that are outside of likely range. This could also fail on road with sharper turns because likely range would widen so it would be harder to discard incorrect estimates.
I could do the following things to make the pipeline more robust:
* Use weighted average between left and right line to estimate lane curvature, assigning weights proportional to the number of detected pixels in each line. Reasoning for this is that we are more confident in estimated parameters there where we had more data points (pixels).
* Try grayscale component in addition to saturation, to see if some gradients complement each other in order to detect more lane line pixels without detecting other things.
