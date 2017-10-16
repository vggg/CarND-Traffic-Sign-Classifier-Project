#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./DataSet%20Distribution%20Graph.png "Visualization"
[image2]: ./grayscale.png "Grayscaling"
[image3]: ./test_images/0.jpeg "Random Noise"
[image4]: ./test_images/1.jpeg "Traffic Sign 1"
[image5]: ./test_images/2.jpeg "Traffic Sign 2"
[image6]: ./test_images/3.jpeg "Traffic Sign 3"
[image7]: ./test_images/4.jpeg "Traffic Sign 4"
[image8]: ./test_images/5.jpeg "Traffic Sign 5"

## Rubric Points


---
###Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/vggg/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because it reduces the data set from rgb to gray which can be represented by 0-127 without much loss of information. 

Here is an example of a traffic sign image before and after grayscaling & histogram normalization.

![alt text][image2]


####2. I used the LeNet model with the input layer (32,32,1) and finaly classification output to 43.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    |Layer 2: Convolutional. Output = 10x10x16    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	|  Input = 5x5x16. Output = 400.			|
| Fully connected		| Input = 400. Output = 120        									|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84.       									|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43      									|
| Softmax				|         									|
|						|												|
|						|												|
 


####3.Trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size of 128, number of epocs 30 and AdamOptimizer.
EPOCHS = 30
BATCH_SIZE = 128


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.931 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the images are not cropped to area of interest, lighting conditions and line of sight might be few of the reasons for not getting the best results.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30k/hr      		| Speed Limit 30k/hr  									| 
| General Caution    			| General Caution  										|
| Priority Road					| Priority Road											|
| Slippery Road      		| 80 kn/hr				 				|
| Road Work			| Road Work      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


