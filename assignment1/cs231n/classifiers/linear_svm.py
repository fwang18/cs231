import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero; 3072*10

  # compute the loss and the gradient
  num_classes = W.shape[1] #W: 3072*10
  num_train = X.shape[0] #X: 5000*3072
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) #1*3072x3072*10=1*10
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i] #only falls on the right class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #print(X)
  margin=(np.dot(X, W).T-np.dot(X, W)[range(X.shape[0]),y]).T+1 #to all, substract right class's score
  margin[range(X.shape[0]),y]=0 #score for right class is 0
  margin = margin * (margin > 0) #if margin<0, replace it with 0; else keep it
  loss=np.sum(margin)/(X.shape[0]) +0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #times_for_each_wj=np.sum(margin>0, axis=1) #margin: 5000*10; this matrix: 10*1
  #np.sum(np.dot(W, times_for_each_wj))
  check_zero=(margin>0).astype(np.int16) #5000*10
  check_zero[range(X.shape[0]),y]-=np.sum(check_zero, axis=1) #total times we substract w_y
  dW=np.dot(X.T, check_zero)/X.shape[0]+reg*W #5000*3072x5000*10=3072*10

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
