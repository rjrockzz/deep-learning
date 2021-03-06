# Basic Notes.
<p align="center">
  <kbd>
  <img src="https://github.com/rjrockzz/deep-learning/blob/master/Coursera/dl2.png">
  </kbd>  
</p><br>

* An image is represented by a 3D array of shape ```(length, height, depth = 3)```. However, when you read an image as the input of an algorithm you convert it to a vector of shape ```(length * height * 3, 1)```. In other words, you "unroll", or reshape, the 3D array into a 1D vector.
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/rjrockzz/deep-learning/blob/master/Coursera/dl.png">
  </kbd>  
</p><br>

* Another common technique we use in Machine Learning and Deep Learning is to **normalize** our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to <img src="https://render.githubusercontent.com/render/math?math=\frac{x}{\| x\|}"> (dividing each row vector of x by its norm).<br>

* [Broadcasting:](https://numpy.org/doc/stable/user/basics.broadcasting.html)  Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. <br><br>
<p align="center">
  <kbd>
  <img src="https://github.com/rjrockzz/deep-learning/blob/master/Coursera/Screenshot%20(188).png">
  </kbd>  
</p><br>

