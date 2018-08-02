**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/exampleimages.png
[image2]: ./output_images/Hist.png
[image3]: ./output_images/HOG.png
[image4]: ./output_images/detection1.png
[image5]: ./output_images/testvid1totalheat.png
[image6]: ./output_images/testvid1.jpg
[video1]: ./project_video.mp4

<!---## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.--->  

---
### Project Code
The project code is given in [Vehicle-Detection.ipynb](https://github.com/anammy/CarND-Vehicle-Detection/blob/master/Vehicle-Detection.ipynb)

### Feature Extraction

#### 1. Histogram of Oriented Gradients (HOG) 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the histogram of oriented gradients (HOG) output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixel_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

I arrived at the above set of HOG parameters to use on all 3 color channels by varying them one by one in different colorspaces and looking at the effects on the visualized HOG representations.

#### 2. Other Features

I also used spatially binned color and color histograms in the `YCrCb` colorspace and concatenated them into the feature vectors. The following parameters were used to extract these features:`spatial_size = (16, 16)`, `hist_bins = 32`, and `hist_range = (0, 256).` The following is an example of the color histograms of car vs. not car images.

![alt text][image2]

#### 3. Classifier

I shuffled and split the dataset 80/20 into training and testing sets. This data was then used to train a linear SVM and generate predictions for the test dataset. The classifier reported an accuracy of 98.79% on the test set.

### Sliding Window Search

#### 1. Window Search

I used the slide window search algorithm within the `find_cars` function provided in the lessons. This function was used to search images with various combinations of scales between 0.5 to 3. In the final iteration, I used two different scales to search the test images and project video.

#### 2. Test Images

Ultimately, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  Here is an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Project Video Analysis
Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Filtering Out False Positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual cells in the heatmap.  I then assumed each cell corresponded to a vehicle. I constructed bounding boxes to cover the area of each cell detected.  

In order to remove false positives, I averaged the heat map from the current frame with the heat maps from the previous 9 frames. I then thresholded this average heat map with a threshold limit of `2 + n//2` where n is the number of previous heatmaps stored. I arrived at this threshold limit through trial and error and defined it dynamically with respect to n in order to accomodate the beginning of the project video.

Here's an example result showing the averaged heatmap from a series of video frames and the bounding boxes then overlaid on the last video frame:

### Averaged heatmap from all 10 frames:
![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]



---

### Discussion

#### 1. Areas of Improvement and Limitations of Car Detection Pipeline

My final output video stream contains some false positives. I would experiment with different classifiers and add the Udacity dataset to my training dataset to see if I can improve the classification accuracy on the project video. I would also explore better filtering techniques to remove false positives. This pipeline currently stores the heat maps of previous iterations and uses them to threshold out the false positives. Instead of storing heat maps, I would explore using the bounding boxes found in the previous iterations to add heat to my current heat map. This might make my pipeline more rubost in eliminating false positives and improve the bounding box boundaries to fully encompass the vehicle. It would also be more efficient since it would require storing a list of tuples instead of a list of image sized arrays from previous iterations.

I would also try to modify the training set with image operations in order to make time series images of cars look different from frame to frame. This would help prevent overfitting of the classifier.

The current algorithm probably cannot handle roads with lots of traffic as the heat map would become roughly uniform. Extracting bounding boxes for individual vehicles from a roughly uniform heat map would be difficult. 



