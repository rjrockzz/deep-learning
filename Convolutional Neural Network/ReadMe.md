# 💫Convolutional Neural Network:

### 👁️‍🗨️ Convolution:
- In mathematics, casually speaking, a mixture of two functions. In machine learning, a convolution mixes the convolutional filter and the input matrix in order to train weights.<br><br>
- The term "convolution" in machine learning is often a shorthand way of referring to either convolutional operation or convolutional layer.<br><br>
- Without convolutions, a machine learning algorithm would have to learn a separate weight for every cell in a large tensor. For example, a machine learning algorithm training on 2K x 2K images would be forced to find 4M separate weights. Thanks to convolutions, a machine learning algorithm only has to find weights for every cell in the convolutional filter, dramatically reducing the memory needed to train the model. When the convolutional filter is applied, it is simply replicated across cells such that each is multiplied by the filter.<br>
<p align="left">
  <kbd> 
    <img width="430" height="300" src="https://media3.giphy.com/media/i4NjAwytgIRDW/200.gif"> 
    <img width="430" height="300" src="https://i.pinimg.com/originals/95/b5/2d/95b52d82200da8ba0ed4615273da474e.gif"> 
</kbd> 
</p><br>

### 👁️ Convolutional filter:
- One of the two actors in a convolutional operation. (The other actor is a slice of an input matrix.) A convolutional filter is a matrix having the same rank as the input matrix, but a smaller shape. For example, given a 28x28 input matrix, the filter could be any 2D matrix smaller than 28x28.<br><br>
- In photographic manipulation, all the cells in a convolutional filter are typically set to a constant pattern of ones and zeroes. In machine learning, convolutional filters are typically seeded with random numbers and then the network trains the ideal values.<br>

### 💥 Convolutional Layer:
- Layer of a deep neural network in which a convolutional filter passes along an input matrix. 
- The following animation shows a convolutional layer consisting of 9 convolutional operations involving the 5x5 input matrix. Notice that each convolutional operation works on a different 3x3 slice of the input matrix. The resulting 3x3 matrix (on the right) consists of the results of the 9 convolutional operations:<br>
<p align="center">
  <kbd> 
    <img width="350" height="200" src="https://developers.google.com/machine-learning/glossary/images/AnimatedConvolution.gif"> 
    </kbd> 
</p><br>

### 🏴‍☠️ Convolutional Neural Network:
- A neural network in which at least one layer is a convolutional layer. 
- A typical convolutional neural network consists of some combination of the following layers:

  - Convolutional layers
  - Pooling layers
  - Dense layers<br>
  
