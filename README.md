#**Traffic Sign Recognition**
###**Report by**:  Soheil Jahanshahi 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist-data.png 
[image2]: ./examples/sample_img_test.png 
[image3]: ./examples/norm_data.png 
[image4]: ./examples/internet_img.png 
[image5]: ./examples/1.png 
[image6]: ./examples/2.png 
[image7]: ./examples/3.png 
[image8]: ./examples/4.png 
[image9]: ./examples/5.png 


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/solix/Traffic_Sign_Deep_Neural_German/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set and shape :

* The size of training set is ?  34799  
* The size of validation set is? 4410 
* The size of test set is ? 12630 
* The shape of a traffic sign image is ? 32 x 32 x 3  (Width, Heigth, Channels)
* The number of unique classes/labels in the data set is ? 43 classes

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the training and validation samples are distributed throughout class IDs. One thing to note is that data is skewed, there are more samples for certain classes between 8-12 that has more data samples than 20-25 for example. this might cause the model to generalize wrongly for classes that has fewer data. The blue color shows distribution of training data and orange color indicates distribution for validation data set.  

![alt Image text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the from fifth cell to tenth code cell of the IPython notebook.

As a first step on fifth code cell of the IPython notebook, I decided to start by shuffling the data and labels to make it more non-deterministic.

As a second step on sixth cell of the IPython notebook, I started by visualizing some sample data from ```X_train``` dataset to get a feeling of how training images and their corresponding label look like. This helped me to see what type of preprocessing operations is needed to apply on images before feeding it into model. Here is an example:

![alt Image text][image2]


Images above are somewhat darker, this might cause some inaccuracy in when training the model, as a third step on 7th to 9th cell ,I applied exposure rescaling to make the images bright enough before feeding it into the model, here is an example after preprocessing the image, you can see that darker images are much brighter now.

![alt text][image3]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook. This code is provided by udacity splitting it into train, test and valid datasets. 

My observation after plotting data was that these traffic data is very skewed. 

To cross validate my model, by keeping test test unseen, untill I have trained my model with high accuracy rate.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

During start of the project I tried to augment some more data using rotate , transform, change light intensity and then populating newly created data into the missing data on some classes using loop. It was a very costly processing task on AWS since I set around 800 images as my upper limit for augmenting. However after training the model I saw a huge decrease in validation accuracy. Thats why I removed the augmenting section from my notebook sheet to focus more on constructing my convolutional layers and hyper parameters. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

I have used NMINST LeNet convolutional network model as an starter template and changed the dimension of iput tensor to fit my input image.  There are two convolutional layer followed by three fully connected layer, the output matches are number of classes for the data i.e. 43 . 


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| Flatten               | output 400
| Fully connected		| output 120       									|
| RELU					|												|
| Dropout				| keep_prop 0.75        									|
| Fully connected		| output 84      									|
| RELU					|												|
| Dropout				| keep_prop 0.75  
| Fully connected		| output 43      									|
 
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper-parameters such as learning rate.

The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

The code for training the model is located in the eighth cell of the ipython notebook and hyper parameters can be found on cell eleven, twelve and fourteen. 

For an optimizer I used Adam optimizer, I used it because it suggest less hyper parameter tuning and also suggested by lecturer at tenserflow lab so I read some paper out of curiosity to find out more advantages:
> The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning.
>
> -- <cite>Diederik P. Kingma, Jimmy Ba - Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015 </cite>

To train the model, I used batch size of 128 with 40 number of epochs, this number achieved by trial and error, meaning that after trying 100 epochs after 40 steps the accuracy wasn't increased and start iterating in ascending/descending order. This batch size has helped me to achieve satisfiable accuracy. slow learning rate of 0.0009 has been chosen because It makes the model to be training slower and also indicated in the lecture keep calm and reduce your learning rate when things go wrong.

My training pipeline can be found in line twenty-two till twenty-fourth of jupyter notebook. pipeline:

* ```logit``` to get the output of the conv network
* ```cross entropy``` to apply to logit for calculating the loss
* ```Adam Optimizer``` for calculating gradient and backpropagation in order to minimize the loss function.
* In the end of the pipeline, evaluate model accuracy using evaluate helper function borrowed from LeNet lab.
   

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twenty-third and twenty-fourth cell of the Ipython notebook.

My final model results were:

* validation set accuracy of ? 97.0%
* test set accuracy of ? 94.8%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? <br />
I started by preprocessing the image to get a brighter version , and after building my LeNet template model I ran my model with 10 epochs and 128 batches with learning rate of 0.001. The reason I chose these numbers because it was shown on the video form the lecture that it gives a satisfying accuracy rate, so I wanted to build up my model from there.
* What were some problems with the initial architecture?  <br />
	One of the problem I encountered was that there was  no increase in accuracy after 5 epochs so from there I figured that I need a to start by changing the hyper-parameters to fine tune the model first. I also had spent lots of time learning how to augment data which went wasted because I suspect I didn't gave a right hyper-parameters for rotation and exposure. it drastically decreased my accuracy model and since it was a very costly task I left it aside due to time limitation. 
* How was the architecture adjusted and why was it adjusted? Which parameters were tuned? How were they adjusted and why? <br/>I applied slower learning rate to let the network learn in slower pace, also increased my epochs to 40 this has boosted my accuracy from 89% to 92% . Thereafter I have adjusted sigma value for weights layers to 0.03 that really came from trial and error. After applying these processes I could achieve 97%. Many of these approaches are given during the lectures and I tried to apply that so that I comprehend better how the model reacts to changes. I have done fine tuning by changing one hyper-parameter at a time. 

  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? <br/>
 	One of my important decision was to take care of my data to be non deterministic. In order to avoid overfitting I used dropout of 0.75 for my training data and 1.0 for my validation, testing and example images data.  

If a well known architecture was chosen:

* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? <br /> I chose LeNet because It was proven to me that it works for recognising MNIST data.  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? <br /> data is splitted to three portion,I use the training portion to train the model and let the model learn about interesting characteristic on its own, and I evaluated how well the model does by showing unseen validation portion. finally after I was done with training my model and satisfied with accuracy rate, I ran my test set(which is a complete unseen data) for the model to predict new images. all accuracy rate were high enough to be proven that model is doing well enough! 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web , reshaped and preprocessed:

![alt text][image4] 

Unfortunately the result of my internet images indicates awful accuracy, I have spent alot of time on preprocessing images, converting it to float32 and reshaping it so that I can get a better prediction, but still images were not comparable with test data. One of the observations was that the German data is dark and noisy where internet images are bright and has a light background. this alone brought my suspicion that my model is not going to predict good on these data. 
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twenty-secondth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(20km/h)      	| Speed limit (70km/h)   									| 
| Speed limit(30km/h)     			| Speed limit (70km/h)										|
| Right of way at the next intersection					| Speed limit (70km/h)											|
| priority road	      		| Speed limit (70km/h)					 				|
| Beware of ice/snow			| Slippery Road      							|


The model was able to correctly guess none of the 5 traffic signs, which gives an accuracy of 0%. This compares infavourably to the accuracy on the test set of 94.8%. I strongly think that there are two main reasons that guesses were incorrect for new images: 

* my model needed more data for classes with less data to generalize better on new images
* I am missing a skill to correctly preprocess the internet images so that it is well prepared for evaluating.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23th and 24th cell of the Ipython notebook.

the result of the prediction is showing that model tend to predict 4 out of five images as class 4 that is speed limit 70 km, looking at the number of test data at this class indicates that model genralize towards skewed part that is data with more number of examples. 

for the first image probability: <br/>
* top prediction= 0.02329394 ,  top class = 4     
![alt text][image9] 

for the second image probability:  <br/>
<br/>
* top prediction= 0.02327329 ,  top class = 4  
![alt text][image8] 

for the third image probability:  <br/>
* top prediction=  0.02328351 ,  top class = 4  
  
![alt text][image7] 

for the fourth image probability:   <br/> 
* top prediction= 0.02327105 ,  top class = 4   
![alt text][image6] 

for the fifth image probability:  <br/>
* top prediction= 0.02329117 ,  top class = 30   

![alt text][image5] 


obviously you can see the the top max probability is very low and distributed evenly between class as rate 0.02xx which is a very low probability.
### Self Reflection and Improvements

This project was an awesome way for me to demystify deep learning from novice to intermediate. Also improved my python programming skills along the way. I could spend more time improving my model so that it recognises new images with high confidence, but I have to continue with next project and lessons. I have challenges managing my time next to my work and that brought some stress and influenced my choices completing this project. 

some of the recommendation to improve the network:

* Augment data
* Add more convolutional layers with same padding and stride 1 helps network to keep more information of input and generalize better.
* L2 regularization would also be an asset. 
* doing more in-depth analysis of network by visualizing the layers and see which interesting part the layers will count as interesting features. 