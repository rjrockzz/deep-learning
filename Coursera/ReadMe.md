# Basic Notes.

* An image is represented by a 3D array of shape ```(length, height, depth = 3)```. However, when you read an image as the input of an algorithm you convert it to a vector of shape ```(length * height * 3, 1)```. In other words, you "unroll", or reshape, the 3D array into a 1D vector.
<br>
<p align="center">
  <kbd>
  <img src="https://github.com/rjrockzz/deep-learning/blob/master/Coursera/dl.png">
  </kbd>  
</p><br>

* Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to <img src="https://render.githubusercontent.com/render/math?math=\frac{x}{\| x\|}"> (dividing each row vector of x by its norm).

For example, if For example, if <img src="https://render.githubusercontent.com/render/math?math=$$x = 
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}\tag{3}$$ then $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix}\tag{4} $$and        $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}\tag{5}$$>
