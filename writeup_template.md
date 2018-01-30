# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image3]: ./examples/center_road_tk1.jpg "Center Road"
[image4]: ./examples/recovery_lap_CounterClockwise_tk1.jpg "Recovery Track 1 counter-clockwise"
[image5]: ./examples/recovery_lap_left_tk1.jpg "Recovery Track 1 Left Camera"
[image6]: ./examples/recovery_lap_tk1.jpg "Recovery Track 1 Center Camera"
[image7]: ./examples/MeanSquaredErrorLoss_8epochs.png "Mean Squared Error Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers 

The model includes RELU layers to introduce nonlinearity (model.py code line 166-170), and the data is normalized in the model using a Keras lambda layer (model.py code line 163). 

#### 2. Attempts to reduce overfitting in the model

There are no dropout layers required to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 148). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 186).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:	
	Recording while the car is running on the center of the road (track 1 and track 2)
	Recording while the car running on the center of the road counter-clockwise (track 1)
	Recover manoeuvres (track 1)
	Recover manoeuvres if the car veers off to the side (track 1) 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the weights of my network to minimize the mean-squared error between the steering command output by the network, and either the command from the recorded data or the adjusted steering command for off-center and rotated images

My first step was to use the NVIDIA convolution neural network model as it proved to be simple and a small number of training epochs could do the job just well around the first track.
There is no overfitting issue if the number of training epochs is low (below 8 from the evidence I have).However 3 epochs shall be enough to have a good performing run around the 1st track.
Therefore no need of dropout or max pooling layers while the number of training epochs is kept within one digit.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

Adding a Cropping2D Layer proved to be useful for choosing an area of interest that excludes the sky.

The lambda layer was a convenient way to parallelize image normalization as well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I recorded more recovery laps when the car veers off to the side and 
a separate counter-clockwise recovery lap as well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving and another one counter-clockwise. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer when there is an offset steering measurement provided by the network. 
These images show recovery manoeuvres:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles which proved to be a useful data augmentation technique which enhanced the dataset.

After the collection process, I had 33808 number of training data points. 
I preprocessed this data by extracting the folders path and then parsing the row contents from the driving log csv file.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the model history. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image7]