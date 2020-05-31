# 🧠 Deep Learning<br>

### Gradient Descent Optimization Algorithms.
<p align="left">
  <kbd>
  <img width="430" height="400" src="https://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif">
    <img width="430" height="400" src="https://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif">
  </kbd>  
</p><br>

* As we can see, the adaptive learning-rate methods, i.e. Adagrad, Adadelta, RMSprop, and Adam are most suitable and provide the best convergence for these scenarios.<br>
* In image 1, Note that Adagrad, Adadelta, and RMSprop almost immediately head off in the right direction and converge similarly fast, while Momentum and NAG are led off-track, evoking the image of a ball rolling down the hill. NAG, however, is quickly able to correct its course due to its increased responsiveness by looking ahead and heads to the minimum.<br>
* In image 2, behaviour of the algorithms at a **saddle point**, i.e. a point where one dimension has a positive slope, while the other dimension has a negative slope, which pose a difficulty for SGD as we mentioned before. Notice here that SGD, Momentum, and NAG find it difficulty to break symmetry, although the two latter eventually manage to escape the saddle point, while Adagrad, RMSprop, and Adadelta quickly head down the negative slope.
----------------
### Conclusions:
* **RMSprop** is an extension of Adagrad that deals with its radically diminishing learning rates.
* It is identical to **Adadelta**, except that **Adadelta** uses the RMS of parameter updates in the numinator update rule.
* **Adam**, finally, adds bias-correction and momentum to RMSprop.
* **RMSprop**, **Adadelta**, and **Adam** are very similar algorithms that do well in similar circumstances.
* **Adam** might be the best overall choice.
<br>*Note:*<br><br>
Interestingly, many recent papers use vanilla SGD without momentum and a simple learning rate annealing schedule. As has been shown, SGD usually achieves to find a minimum, but it might take significantly longer than with some of the optimizers, is much more reliant on a robust initialization and annealing schedule, and may get stuck in saddle points rather than local minima. Consequently, if you care about fast convergence and train a deep or complex neural network, you should choose one of the adaptive learning rate methods.


