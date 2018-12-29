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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    score = X[i].dot(W)
    exp_score = np.zeros_like(score)
    exp_sum = 0
    #compute loss; y[i] represente the correct class
    for j in xrange(num_classes):
        exp_score = np.exp(score[j])
        exp_sum += exp_score
    loss += (-1)*np.log(np.exp(score[y[i]]) / exp_sum)
    
    #compute dW
    for k in xrange(num_classes):
        if k != y[i]:
            dW[:,k] += (np.exp(score[k]) / exp_sum) * X[i]
        else:
            dW[:,k] += ((np.exp(score[k]) / exp_sum) - 1) * X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  score = np.dot(X,W)
  #!!!attention: Numeric stability.
  
  max_score = np.max(score, axis = 1).reshape(num_train, 1)
  score -= max_score
  correct_score = score[np.arange(num_train), y[:]].reshape(num_train, 1)
  exp_score = np.exp(score)
  exp_sum = np.sum(exp_score, axis = 1).reshape(num_train, 1)
  loss += np.sum(np.log(exp_sum) - correct_score)
  
  #shift_scores = score - np.max(score, axis = 1).reshape(-1,1)
  #print(np.max(score, axis = 1).shape)
  #softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  #loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
    
  #compute dW
  margin = exp_score / exp_sum # N x C
  margin[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, margin)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

