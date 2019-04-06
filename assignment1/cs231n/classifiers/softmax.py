import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #if i < 20 :
    #    print("scores ",scores)
    max = scores.max()
    scores -= max
    sum = np.sum(np.exp(scores))
    correct = np.exp(scores[y[i]])
    loss += - np.log(correct/sum)
    scores[y[i]] -= 1
    for j in range(num_classes) :
        if j == y[i] :
            dW[:,j] += -1 * ((sum - correct) / sum) * X[i]
        else :
            dW[:,j] += (np.exp(scores[j])/sum)*X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= scores.max()
  scores = np.exp(scores)
  correct = np.zeros(y.shape)
  correct = scores[range(scores.shape[0]),y]
  sum = np.sum(scores,axis=1)
  #print("sum shape ", sum.shape)
  #print("y shape ", y.shape)
  loss = np.sum(-1 * np.log(correct/sum))

  tmp = np.divide(scores, sum.reshape(num_train,1))
  tmp[range(scores.shape[0]),y] = -1 * ((sum - correct)/ sum)
  dW = X.T.dot(tmp)
  #print("correct shape ", correct.shape)
  #print("score ",scores[: 5][: 5])
  #print("y ",y[: 5])
  #print("correct ",correct[: 5])

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

