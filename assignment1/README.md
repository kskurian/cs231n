

# Environment setup
<br/>
I. Locally <br/>
1. Installing Anaconda:If you decide to work locally, we recommend using the free Anaconda Python distribution, 
which provides an easy way for you to handle package dependencies <br/>
2. Anaconda Virtual environmentTo set up a virtual environment>conda create -n cs231n python=3.6 anaconda <br/>
3. Then, to activate and enter the environment, run>source activate cs231nto exit>source deactivate cs231n <br/>

To activate this environment, <br/>
use:# > activate cs231n## <br/>

To deactivate an active environment, <br/>
use:# > deactivate <br/>

To start the Jupyter Notebook, we need to first move to the Anacoda3 folder. <br/>
> cd C:\Users\<name>\Anaconda3\envs>jupyter notebook <br/>

$ wget http://cs231n.github.io/assignments/2018/spring1718_assignment1.zip 
$ sudo apt-get install unzip <br/>
$ unzip spring1718_assignment1.zip <br/>
$ cd assignment1$ ./setup_googlecloud.shII.  <br/>
<br/>

II. GOOGLE CLOUD : <br/>
>gcloud compute ssh --zone=<instance> 
>source .env/bin/activate>jupyter notebook --ip=0.0.0.0 --port=7000 --no-browserconnect 
<br/>
via web browser in http, :7000 with the passcode <br/>
This has to be done since in the gcp console you have given the ip configuration for 0.0.0.0:7000. <br/>
Else it wouldn't work. <br/>
There is another way as well by using ssh port  ref : https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/ <br/>

ASSIGNMENT 1 : <br/>

The whole problem is based on Image Classification of the data. Here we are trying to use   <br/>
1. KNN  <br/>
2. Linear Classification  <br/>
3. Neural Networks  <br/>

********************************************************************************************************************************
# About the dataset - 
CIFAR http://www.cs.toronto.edu/~kriz/cifar.html

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle.Loaded in this way, each of the batch files contains a dictionary with the following elements:
	* data -- a 10000x3072 numpy array of uint8s.  <br/>
	Each row of the array stores a 32x32 colour image.  <br/>
	The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
	The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
	of the first row of the image.
	* labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith
	image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
	* label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above.
	For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

*******************************************************************************************************************************
# KNN

It tries to create a division of the point based on the distance this can being any distance such Euclidean etc.  <br/>
It take a set of training data as the true and   <br/>
print('Training data shape: ', X_train.shape)   <br/>
(50000, 32, 32, 3)  <br/>
print('Training labels shape: ', y_train.shape)  <br/>
(50000,)  <br/>
print('Test data shape: ', X_test.shape)  <br/>
(10000, 32, 32, 3)  <br/>
print('Test labels shape: ', y_test.shape)  <br/>
(10000,)  <br/>

We shorten the X_train and Y_train to 5000 and x_test and y_test to 500.  <br/>
We next find the distance using Euclidean method.  <br/>
Performance improvement happens as we make the process more matrix based once  <br/>
* Two loop version took 61.402659 seconds  <br/>
* One loop version took 54.067250 seconds  <br/>
* No loop version took 0.588316 seconds  <br/>

Validation sets for Hyperparameter tuning  <br/>
We cannot use the test set for the purpose of tweaking hyperparameters, you should think of the test set as a very precious resource that should ideally never be touched until one time at the very end. Only use the test set once at end, it remains a good proxy for measuring the generalization of your classifier. Else you would overfit to the test set.

Cross-validationIn cases  <br/>
where the size of your training data might be small use more sophisticated technique for hyperparameter tuning called cross-validation. We split the training set into folds, keeping the last set as validation set. Now we train the model with each of these and then run the predict with the validation set. Here we test the accuracy across different training set.

*******************************************************************************************************************************

# SVM - support vector machineIn this exercise you will:
	* implement a fully-vectorized loss function for the SVM
	* implement the fully-vectorized expression for its analytic gradient
	* check your implementation using numerical gradient
	* use a validation set to tune the learning rate and regularization strength
	* optimize the loss function with SGD
	* visualize the final learned weights

Split the data into train, val, and test sets.   <br/>

In addition we will create a small development set as a subset of the training data;  <br/>
 we can use this for development so our code runs faster. <br/>
 print('Train data shape: ', X_train.shape) <br/>
 #(49000, 32, 32, 3) <br/>
 print('Train labels shape: ', y_train.shape) <br/>
 #Train labels shape: (49000,) <br/>
 print('Validation data shape: ', X_val.shape) <br/>
 #Validation data shape: (1000, 32, 32, 3) <br/>
 print('Validation labels shape: ', y_val.shape) <br/>
 #Validation labels shape: (1000,) <br/>
 print('Test data shape: ', X_test.shape) <br/>
 #Test data shape: (1000, 32, 32, 3) <br/>
 print('Test labels shape: ', y_test.shape) <br/>
 #Test labels shape: (1000,) <br/>

 X_Training data shape: (49000, 3072) <br/>
 X_Validation data shape: (1000, 3072) <br/>
 X_Test data shape: (1000, 3072) <br/>
 X_dev data shape: (500, 3072) <br/>
 
 # Preprocessing:  <br/>
 subtract the mean image <br/>
 first: compute the image mean based on the training data <br/>
 second: subtract the mean image from train and test data <br/>
 third: append the bias dimension of ones (i.e. bias trick) <br/>
 
 so that our SVM only has to worry about optimizing a single weight matrix W. <br/>
 After the third preprocessing : <br/>
 X_train.shape (49000, 3073) <br/> 
 X_val.shape   (1000, 3073) <br/>
 X_test.shape  (1000, 3073) <br/>
 X_dev.shape   (500, 3073) <br/>
 
 The prediction is given by :f(xi,W,b)=Wxi+b <br/>
 Where W is the weights and b is the bias. <br/>
 For this to got we use the training set and get the values of W and b. <br/>
 The value of W is first set randomly and then against each run on the <br/>
 training we check a value know as Loss. We try decrease the loss and <br/>
 modify our W, the loss can be computed using many means and the one <br/>
 we are using is multi-class SVM loss function. <br/>
 Once the a loss function is decided we see how to optimize it and  <br/>
 thats the gradient descent does. <br/>
 We use the optimization to lower our loss. <br/>
 
 Multiclass Support Vector Machine :  <br/>
 Loss Function= <br/>
 (1/N) Sum over all the rows [ Sum of the different classes except for j == y[i] [ MAX( 0, f(xi;W)j−f(xi;W)yi+1) ]+α R(W) <br/>
 R(W) = W*W which is used to normalize weight values.  <br/>
 :: Regularization Loss <br/>
 Here the Delta is 1 a hyperparameter but usually used as 1. This is the difference between correct class and the wrong on, ie minimum difference.
 
 1. We need to get the Loss and know the direction we could progress to decrease the loss.
 This is done by finding the gradient descent which is the differential of the loss function.
 The Gradient Function for SVM since its max function we cant take direct differential,
 Hence we use based on the boundary  dL/dW since the function of Loss is based on the Weight hence
 we differentiate based on the Weight.
 
 We use the concept of Indicator function ie for a condition the function returns a value which is fixed.
 Here we are using a 1 indicator function.
 
 As we see here the Loss function is the average is 
 1/N and sum of alpha * Sum square of Weights.
 Once we find the loss and the gradient descend we need to update the values for the Linear_Classifier.
 While prediction we need to update the weights using the learning rate.

*Huge remark is the that the regularization value was the mistake. Which made the computation make loss at nan.

*******************************************************************************************************************************
# Softmax - 
	This exercise is analogous to the SVM exercise. You will:
	* implement a fully-vectorized loss function for the Softmax classifier
	* implement the fully-vectorized expression for its analytic gradient
	* check your implementation with numerical gradient
	* use a validation set to tune the learning rate and regularization strength
	* optimize the loss function with SGD
	* visualize the final learned weights
	
The large part of Softmax uses the same as SVM since the predict and train is the linear classifier just the difference is in using the loss function.
Similarly which would change the gradient descent. The accuracy checking and getting the best instance of the model is also the same.

The binary Logistic Regression classifier  which is knownbefore, the Softmax classifier is its generalization to multiple classes. 

The Loss function is log( Pi) = log( exp(fi)/Sum of exp fj ) = log(exp( Xi.W[y[i]] / Sum of exp (Xi.W) ))
The loss value is the sum of the all these and a single value.
This sum could be a huge value if we directly take the exp of all the value. So to make the sum smaller we subtract with the highest value across all.
	X - N,D
	Y - N,
	W - D,C
	scores - N,C
	dW - D,C where for incorrect exp
	The gradient descent is a matrix on D,C - number of parameters, classes. pk -1 

************************************************************************************************
	The SVM classifier uses the hinge loss, or also sometimes called the max-margin loss. 
	The Softmax classifier uses the cross-entropy loss. 
	The Softmax classifier gets its name from the softmax function, which is used to squash 
	the raw class scores into normalized positive values that sum to one,
	so that the cross-entropy loss can be applied.

*******************************************************************************************************************************
  # Two Layer Nueral Network -  <br/> <br/>
  X: (N, D) <br/>
  y: Vector of training labels <br/>
  W1: (D, H), First layer weights;  <br/>
  b1: (H,) First layer biases; <br/>
  Intermediate result : (N,H) after relu activation <br/>
  W2: (H, C) Second layer weights;  <br/>
  b2: (C,), Second layer biases; has shape  <br/>
  scores : (N,C) <br/>
  If y is None, return a matrix scores of shape (N, C) <br/>
   <br/>
  ![Image of Front](https://github.com/kskurian/cs231n/blob/master/assignment1/Ml.png?raw=true)
  dW1: H,C x C,N x N,D would be the same as that of W1 : D,H  <br/>
  db1: N,C * C, H = N, H ->  <br/>
  dW2: H,N * N,C = H,C -> Since its derivative is Out of Hidden X dscore <br/>
  db2: (C,) -> Since its derivative is 1. Its simply 1 X dscore <br/>
  dscore : ( N,C ) <br/>
  
  Here we are using an extra hidden layer and the activation function of Relu <br/>
  Using ReLUs as the Activation Function : The relu simply takes the positive part of any argument  <br/>
  ie its function is x+. Another possible function that can be compared is the Sigmoid function. <br/>
  Hence we would take either 0 or the positive value. <br/> <br/>
  
  After calculating the scores we calculate the loss by using softmax classifier for the given scores. <br/>
  loss = data_loss + reg_loss <br/>
  Here we can get the data loss from using the softmax classifier and for the regularization  <br/>
  we need to consider both the weights that is being used. <br/>
  
  Hence in the forward passing the first output would be taking the input and then Weight + bias.  <br/>
  This is the first layer then we have the hidden layer which is second layer.  <br/>
  The first activation layer uses the Relu Activation which is positive value of the transformations. <br/>
  Then the second value is again with W2 and b2 bias is added. <br/> <br/>

  We calculate with the softmax classifier for the loss and gradient descent. <br/>
  In fact the gradient descent found is basically the same dW2 and  <br/>
  the multiplication is the derivative dF/dW2 = The Layer1 output (X.W1 + b1) <br/>
  multiplied by Derivative of the Function with respect to input. <br/>
  Hence this is same as that of the back propagation. <br/> <br/>
  
  The validation accuracy isn't very good. <br/>
  One strategy for getting insight into what's wrong is to plot the loss function and the accuracies <br/>
  on the training and validation sets during optimization. <br/>
  This clearly shows us that how much over fitting is happening to the model. Now we modify the value to get <br/>
  better prediction of the images. <br/>
    <br/> <br/>
  Another strategy is to visualize the weights that were learned in the first layer of the network. <br/>
  In most neural networks trained on visual data, the first layer weights typically show some visible <br/>
  structure when visualized. This is similar to the template structure used to visualize in the linear classifier. <br/>
   <br/> <br/>
  Now so we need to tune our hyper-parameters . <br/>
  The hyper-parameters that we have here are :  <br/>
  	1. Hidden Size : which is H the size of Weight Created <br/>
  	2. Learning Rate : which is the descent that we are having in Gradient Descent <br/>
  	3. Regularization : The figurative value added to decrease the Weight being higher <br/>
   <br/> <br/>
  Looking at the visualizations above, we see that the loss is decreasing more or less linearly, <br/>
  which seems to suggest that the learning rate may be too low.  <br/>
  Moreover, there is no gap between the training and validation accuracy, suggesting that the model  <br/>
  we used has low capacity, and that we should increase its size. On the other hand, with a very large model. <br/>
  We would expect to see more overfitting, which would manifest itself as a very large gap  <br/>
  between the training and validation accuracy. <br/>
   <br/> <br/>
  Once we train we would have <br/>
        { 'loss_history': loss_history, <br/>
        'train_acc_history': train_acc_history, <br/>
        'val_acc_history': val_acc_history } <br/>
  
  Using these we try find the right value of our hyper-parameters. <br/>
  ref :  http://cs231n.stanford.edu/vecDerivs.pdf
