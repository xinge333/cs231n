import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################

  next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b) # N x H matrix
  cache = (x, prev_h, Wx, Wh, b, next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  (x, prev_h, Wx, Wh, b, next_h) = cache
  N, D = s.shape, H = prev_h.shape[1]
  # derivative tanh'(s) = 1 - (tanh(s))^2  , here next_h is tanh(s)
  temp = np.ones((N, H)) - np.square(next_h)
  dtanh = np.multiply(dnext_h, temp)
  db = np.sum(dtanh, axis = 0)
  db = db.T #(H,)
  dx = dtanh.dot(Wx.T)
  dWx = x.T.dot(dtanh)
  dprev_h = dtanh.dot(Wh.T)
  dWh = prev_h.T.dot(dtanh)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape, H = h0.shape[1]
  # cache should contain (x, prev_h, Wx, Wh, b, next_h)
  prev_h = h0.copy()
  prev_hs = np.empty([N, T, H])
  next_hs = np.empty([N, T, H]) # 延迟一个时间点的隐状态
  caches = np.empty([N, T, H]) # 保存T时间内的全部cache[-1]这一项
  
  for t in range(T):
    next_h, cache_temp = rnn_step_forward(x[:,t,:], prev_h, Wx, Wh, b)
    prev_hs[:,t,:] = prev_h
    prev_h = next_h
    next_hs[:,t,:] = next_h
    caches[:,t,:] = cache_temp[5]
  
  h = next_hs
  cache = (x, prev_hs, Wx, Wh, b, caches) # Wx, Wh, b is commonly shared
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  x, prev_hs, Wx, Wh, b, caches = cache
  N, T, H = dh.shape
  D = x.shape[2]
  dx = np.empty((N,T,D))
  dprev_h_t = np.zeors((N,H))
  dWx 2 np.zeors((D,H))
  dWh = np.zeos((H,H))
  db = np.zeors(H)

  dh_now = np.empty((N,H)) # hidden state's gradient at now
  
  # reversed time step !!!
  for t in range(T-1,-1,-1):
    dh_now += dh[:,t,:]
    cache_temp = (x[:,t,:], prev_hs[:,t,:], Wx, Wh, b, caches[:,t,:])
    dx_temp, dprev_h_temp, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dh_now, cache_temp)
    dx[:,t,:] = dx_temp
    #shared parameters' gradient should be added up
    dWx += dWx_temp
    dWh += dWh_temp
    db += db_temp
    dh_now = dprev_h_temp
  dh0 = dh_now
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  N, T = x.shape
  out = np.empty((N, T, D))
  for i in range(N):
    for j in range(T):
      out[i,j,:] = W[x[i,j],:]
  cache = (x, W)             
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  N, T = x.shape
  dW = np.zeros_like(W)
  
  for i in range(N):
    for j in range(T):
        dW[x[i,j],:] += dout[i,j,:]
  
  # above for-loops equas to blow expresison
  #np.add.at(dW, x, dout)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  temp = x.dot(Wx) + prev_h.dot(Wh) + b
  ii,ff,oo,gg = np.split(temp, 4, axis = 1)
  i = sigmoid(ii)
  f = sigmoid(ff)
  o = sigmoid(oo)
  g = np.tanh(gg)
  next_c = np.multiply(f, prev_c) + np.multiply(i, g)
  next_h = np.multiply(o, np.tanh(next_c))
  cache = (x, prev_h, prev_c, Wx, Wh, b, temp, next_c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, Wx, Wh, b, temp, next_c = cache
  ii,ff,oo,gg = np.split(temp, 4, axis = 1)
  i = sigmoid(ii)
  f = sigmoid(ff)
  o = sigmoid(oo)
  g = np.tanh(gg)
  #first: caculate dE/dc_(t-1)
  temp1 = np.multiply(dnext_h, o)
  dtanh_ct = np.ones_like(next_c) - np.square(np.tanh(next_c))
  temp2 = np.multiply(dtanh_ct, f)
  dprev_c = np.multiply(temp1, temp2) + np.multiply(dnext_c, f)
  
  # caculate dE/temp
  temp3 = np.multiply(dnext_h, o)
  temp4 = np.multiply(dtanh_ct, temp3)
  df_temp = np.multiply(dnext_c, prev_c) + np.multiply(temp4, prev_c)
  di_temp = np.multiply(dnext_c, g) + np.multiply(temp4, g)
  dg_temp = np.multiply(dnext_c, i) + np.multiply(temp4, i)
  do_temp = np.multiply(dnext_h, np.tanh(next_c))
  df = np.multiply(df_temp, np.multiply(f,np.ones_like(f) - f))
  di = np.multiply(di_temp, np.multiply(i,np.ones_like(i) - i))
  dg = np.multiply(dg_temp, np.ones_like(g) - np.square(g))
  do = np.multiply(do_temp, np.multiply(o,np.ones_like(o) - o))   
  dtemp = np.concatenate((di,df,do,dg),axis = 1)
  # caculate dx dprev_h dWx dWh db
  dx = dtemp.dot(Wx.T)
  dWx = (x.T).dot(dtemp)
  dprev_h = dtemp.dot(Wh.T)
  dWh = (prev_h.T).dot(dtemp)
  db = np.sum(dtemp, axis = 0).reshape(b.shape[0],b.shape[1])
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape, H = h0.shape[1]
  # cache should contain (x, prev_h, prev_c, Wx, Wh, b, temp, next_c, next_h)
  prev_h = h0.copy()
  prev_c = np.zeros_like(prev_h)
  prev_hs = np.empty([N, T, H])
  prev_cs = np.empty([N, T, H])
  next_hs = np.empty([N, T, H]) # 延迟一个时间点的隐状态
  next_cs = np.empty([N, T, H])
  temps = np.empty([N, T, H]) # 保存T时间内的全部cache[-1]这一项

  for t in range(T):
    next_h, next_c, cache = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)
    prev_hs[:,t,:] = prev_h
    prev_cs[:,t,:] = prev_c
    prev_h = next_h
    prev_c = next_c
    next_hs[:,t,:] = next_h
    next_cs[:,t,:] = next_c
    temps[:,t,:] = cache_temp[6]
  
  h = next_hs
  # Wx, Wh, b is commonly shared
  cache = (x, prev_hs, prev_cs, Wx, Wh, b, temps, next_cs)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  x, prev_hs, prev_cs, Wx, Wh, b, temps, next_cs = cache
  N, T, D = x.shape, H = Wh.shape[1]
  dx = np.empty(N,T,D)
  dh0 = np.zeros((N,H))
  dWx = np.zeros_like(Wx)
  dWh = np.zeros_like(Wh)
  db = np.zeros_like(b)
  dh_now = np.zeros((N,H))
  dc_now = np.zeros((N,H))
  
  for t in range(T-1,-1,-1):
    dh_now += dh[:,t,:]
    cache_temp = (x[:,t,:], prev_hs[:,t,:], prev_cs[:,t,:], Wx, Wh, b, temps[:,t,:], next_cs[:,t,:])
    dx_temp, dprev_h_temp, dprev_c_temp, dWx_temp, dWh_temp, db_temp = lstm_step_backward(dh_now, dc_now, cache_temp)
    dh_now = dprev_h_temp
    dc_now = dprev_c_temp
    dx[:,i,:] = dx_temp
    dWx += dWx_temp
    dWh += dWh_temp
    db += db_temp
  dh0 = dh_now
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

