# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the German traffic sign data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/frequency_training.png
[image2]: ./img/frequency_validation.png
[image3]: ./img/frequency_test.png 
[image9]: ./img/training_history.jpg
[image4]: ./data/real/14.png
[image5]: ./data/real/19.png
[image6]: ./data/real/21.png
[image7]: ./data/real/27.png
[image8]: ./data/real/28.png

# Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! Here is a link to my [project code](https://github.com/jknagin/Udacity-CarND-P3-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

I used Numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32x32x3).
* The number of unique classes/labels in the data set is 43.

Here is an exploratory visualization of the data set. It is a set of histograms showing the number of examples belonging to each class in the training, validation, and test sets.

![image1]
![image2]
![image3]

### Design and Test a Model Architecture

I normalized the image data to be within the range of -1 to 1 to aid in the optimization. Normalized inputs give the cost function a nicer shape for optimization. Although this is not the case for this project, it is especially true when the values of the components in the feature vector or matrix are of different scales. This can happen, for example, if the feature components have different units.

I did not convert the images to grayscale because I thought that the color information would have been helpful for the network to classify correctly.

My final model consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |              32x32x3 RGB image              |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6  |
|      RELU       |                                             |
| Convolution 7x7 |  1x1 stride, same padding, outputs 28x28x6  |
|      RELU       |                                             |
| Max pooling 2x2 | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| Convolution 5x5 | 1x1 stride, same padding, outputs 10x10x16  |
|      ReLU       |                                             |
| Max pooling 2x2 | 2x2 stride, valing padding, outputs 5x5x16  |
|     Flatten     |                 outputs 400                 |
| Fully connected |                 outputs 120                 |
|      ReLU       |                                             |
| Fully connected |                 outputs 84                  |
|      ReLU       |                                             |
| Fully connected |           outputs n_classes = 43            |
|     Softmax     |        outputs softmax probabilities        |

 
To train the model, I used the Adam optimizer with a learning rate of 0.001 and a batch size of 128 for 20 epochs.

Initially, I sought to reproduce the 89% validation accuracy that the assignment reported with the LeNet model from the lecture slides. Once I was able to reproduce this result as a baseline, I examined the training accuracy to see how the validation accuracy could be improved. If the training accuracy were also too low, then the model suffered from high bias, and would benefit from adding more layers. If the training accuracy were too high, then the model suffered from high variance, and would benefit from regularization methods, such as reducing the model complexity or adding dropout layers.

It turned out that the LeNet architecture's training accuracy after 10 epochs was in the mid 90s percentage. By this point, the validation accuracy had plateaued to 88-89%. I felt that a mid 90s percentage was too low for training accuracy. Thus, I decided to increase the complexity of the model by adding convolutional layers. After some experimentation with different sizes and locations of convolutional layers, I arrived at the model architecture discussed above.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.4%
* test set accuracy of 92.6%

These accuracy percentages indicate that the model is working reasonably well.

Below is a screenshot of the second half of the training, showing the train and validation accuracy after each epoch:

![training history][image9]


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![stop sign][image4] 
![dangerous curve to the left][image5]
![double curve][image6] 
![pedestrians][image7]
![children crossing][image8]

From top left to bottom right, they are:
* Stop sign (class 14)
* Dangerous curve to the left (class 19)
* Double curve (class 21)
* Pedestrians (class 27)
* Children crossing (class 28)

All of these images should be quite straightforward to classify since they are clear and unambiguous. The images are pictures of digital signs, rather than real-world signs, so there are no issues with orientation, lighting, etc.

The code for making these predictions with my final model is located in the cell under the heading "**Predict the Sign Type for Each Image and Analyze Performance**."

Here are the results of the predictions for the five images:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
|      Stop Sign       |     No vehicles      |
| Dangerous curve left | Dangerous curve left |
|     Double curve     |     Double curve     |
|     Pedestrians      |     Pedestrians      |
|  Children crossing   |  Children crossing   |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavorably to the accuracy on the test set of 92.6%.

For the first image, the model is deciding between whether it is a "no vehicles" sign or a "yield" sign, but the image contains a stop sign. The top five softmax probabilities were as follows:

|      Prediction       | Probability  |
| :-------------------: | :----------: |
|      No vehicles      |   0.512618   |
|         Yield         |  0.4873806   |
|         Stop          | 5.958012e-07 |
| Speed limit (30km/h)  | 3.857693e-07 |
| Speed limit (120km/h) | 2.640065e-07 |

For the second image, the model is quite sure it is a "dangerous curve to the left" sign, and the image does contain a "dangerous curve to the left" sign. The top five softmax probabilities were as follows:

|         Prediction          |  Probability  |
| :-------------------------: | :-----------: |
| Dangerous curve to the left |      1.0      |
|        Slippery road        | 5.785166e-18  |
|        Double curve         | 3.5763116e-30 |
|    Speed limit (60km/h)     | 2.0008355e-31 |
|    Wild animals crossing    | 1.2848357e-31 |

For the third image, the model is quite sure it is a "double curve" sign, and the image does contain a "double curve" sign. The top five softmax probabilities were as follows:

|              Prediction               |  Probability  |
| :-----------------------------------: | :-----------: |
|             Double curve              |      1.0      |
|         Wild animals crossing         | 2.3404133e-13 |
|               Road work               | 5.9706295e-14 |
| Right-of-way at the next intersection | 1.564187e-16  |
|         Speed limit (50km/h)          | 1.3159972e-16 |

For the fourth image, the model is relatively sure it is a "pedestrians" sign, and the image does contain a "pedestrians" sign. The top five softmax probabilities were as follows:

|              Prediction               |  Probability   |
| :-----------------------------------: | :------------: |
|              Pedestrians              |   0.7274194    |
|            General caution            |   0.27257892   |
| Right-of-way at the next intersection | 1.6160571e-06  |
|            Traffic signals            | 4.0429583e-08  |
|     Dangerous curve to the right      | 1.36723246e-11 |

For the fifth image, the model is quite sure it is a "children crossing" sign, and the image does contain a "children crossing" sign. The top five softmax probabilities were as follows:

|              Prediction               |  Probability  |
| :-----------------------------------: | :-----------: |
|           Children crossing           |      1.0      |
| Right-of-way at the next intersection | 1.4754323e-10 |
|             Slippery road             | 1.3363426e-16 |
|         Speed limit (30km/h)          | 8.765549e-18  |
|       Road narrows on the right       | 1.6242238e-18 |