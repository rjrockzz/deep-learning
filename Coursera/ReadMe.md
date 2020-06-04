# Basic Notes.

* An image is represented by a 3D array of shape ```(length, height, depth = 3)```. However, when you read an image as the input of an algorithm you convert it to a vector of shape ```(length * height * 3, 1)```. In other words, you "unroll", or reshape, the 3D array into a 1D vector.
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/rjrockzz/deep-learning/blob/master/Coursera/dl.png">
  </kbd>  
</p><br>
