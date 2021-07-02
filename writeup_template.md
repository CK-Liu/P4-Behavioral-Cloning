# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/error.jpg "error"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/before.jpg "Normal Image"
[image7]: ./examples/after.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car **can not** be driven autonomously in my local laptop around the track by executing
```sh
python drive.py model.h5
```
So I finished this project in Udacity's workspace. You can combine my github's writeup and Udacity's workspace together as my submission.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network (CNN) which followed [NVIDIA end to end learning](https://arxiv.org/abs/1604.07316).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. We also augmented the dataset by flipping the images and the steering angles. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected a combination of center lane driving, recovering from the left and right sides of the road, driving in the reverse direction for one lap. I finished this project in Udacity workspace. I use the default training dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was to build and train a CNN to predict the steering angle to drive a car autonomously in the simulator. The goal of this project: the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-95) consisted of a convolution neural network. Here is a visualization of the architecture:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 320x160x3 RGB image   							|
| Cropping2D     	| upper 70 pixels and lower 25 pixels cropped out 	|
| Lambda					|	normalize image											|
| Convolution 5x5	      	| 2x2 stride,  depth 24 				|
| Convolution 5x5     	| 2x2 stride,  depth 36 	|
| Convolution 5x5					|	2x2 stride,  depth 48					|
| Convolution 3x3	     	| 1x1 stride,  depth 64 				|
| Convolution 3x3		| 1x1 stride,  depth 64      			|
| Flatten		|       			|
| Dense			| depth 100       					|
| Dense			| depth 50       					|
| Dense			| depth 10       					|
| Dense			| depth 1       					|

#### 3. Creation of the Training Set & Training Process

My first attempt is to do this project on my local laptop with a Nvidia GTX1660Ti. The simulator is no problem when recording data. I recorder two lap clock-wise, one lap counter clock-wise and recovery from the left and right side of the road. But when I try to run this simulator in autonomous mode, `python drive.py model.h5` can not move the car in the simulator. There are no steering angle outputs. Here is the picture about I failed in my local laptop:

![alt text][image1]

After several failures about moving the car in my local computer, I decided to use Udacity's workspace. It works pretty well. I saved everything in my workspace. This project has been done in Udacity workspace with the default dataset.

Here is some pictures I recorded on my local laptop using training mode in the simulator. I first recorded two laps on track one using center lane driving. Then I recorded driving in the  reverse direction for one lap. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recovery from the side of the road. These images show the car recovery from the right side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images thinking that this would avoid overfitting and get a more generalized model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also try to use left and right cameras data, line 19 to 22 in model.py, but there maybe some bad images in left and right cameras data. It makes the dimension of the training images is not correct. So I decided to use only the center image and flipped images to augment dataset.

After the collection process, I had 16072 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
