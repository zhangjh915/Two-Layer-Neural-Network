# Two Layer Neural Network Classifier
## Description:

A "Two Layer Neural Network Classifier" on cifar10 images written in Python3.

This model uses cifar10 as the dataset.  The dataset can be downloaded following the instrustions in [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Performance:
The final accuracy of the model is 0.554 and the loss and accuracy curves are shown below.

![alt text](/images/loss_acc_curves.png)

## Model Architecture:
The two-layer neural net includes two fully connected layers with Relu and Softmax. The data are fed into the structure through the first fully connected layer, then a Relu function to add non-linearity, followed by another fully connected layer, and a Softmax function at the end. The forward pass can be expressed as follows:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?h&space;=&space;Relu(W_1x&plus;b_1)" title="h = Relu(W_1x+b_1)" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\hat{y}=Softmax(W_2h&plus;b_2)" title="\hat{y}=Softmax(W_2h+b_2)" />
</p>

The softmax function has L2 regularization and is the same as the other repository of [Softmax Classifier](https://github.com/zhangjh915/Softmax-Classifier-on-cifar10).

To train the neural network, there different gradient descent update methods, which are stochastic gradient descent (SGD), SGD with momentum, and RMSprop. SGD is the same as in the [Softmax Classifier](https://github.com/zhangjh915/Softmax-Classifier-on-cifar10). SGD with momentum has the following update relation:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?V_t=\beta&space;V_{t-1}&plus;\alpha&space;\triangledown_wL(W,&space;x,&space;y)" title="V_t=\beta V_{t-1}+\alpha \triangledown_wL(W, x, y)" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?W=W-V_t" title="W=W-V_t" />
</p>

where <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> is the momentum, <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> the learning rate, and <img src="https://latex.codecogs.com/gif.latex?\triangledown&space;L" title="\triangledown L" /> the calculated gradient. 

SGD with momentum typically has better performance than classic SGD. There are two main reasons. First is due to the gradients calculated by a small batch rather than the entire dataset in SGD, causing non-exact derivatives. And the exponentially weighed averages can provide us a better estimate which is closer to the actual derivate than the noisy calculations. The other reason lies in ravines. Ravines are common near local minimas in deep learning and SGD has troubles navigating them. SGD will tend to oscillate across the narrow ravine since the negative gradient will point down one of the steep sides rather than along the ravine towards the optimum. Momentum helps accelerate gradients in the right direction. The figure below illustrate this point (left: classic SGD, right: SGD with momentum).

![alt text](/images/momentum.png)

The RMSprop update algorithm is as follows:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?V_t=\rho&space;V_{t-1}&plus;(1-\rho)\triangledown_w^2L(W,x,&space;y)" title="V_t=\rho V_{t-1}+(1-\rho)\triangledown_w^2L(W,x, y)" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?V_t=-\frac{\alpha}{\delta&space;&plus;V_t}\triangledown_wL(W,x,&space;y)" title="V_t=-\frac{\alpha}{\delta +V_t}\triangledown_wL(W,x, y)" />
</p>

where <img src="https://latex.codecogs.com/gif.latex?\rho" title="\rho" /> is the decay rate, <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> the learning rate, <img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /> a small constant (10<sup>-6</sup> for example), and <img src="https://latex.codecogs.com/gif.latex?\triangledown&space;L" title="\triangledown L" /> the calculated gradient.

The key concept of RMSporp is to control how much history information is to be used by adjusting the parameter of <img src="https://latex.codecogs.com/gif.latex?\rho" title="\rho" />.

The parameters can be further tuned to obtain a better performance.

## Code Usage
1. Clone or download the code
2. Create a folder called "data" and unzip the downloaded cifar10 data in it
3. Run model_visualization.py

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/).
2. [https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d).
3. [https://blog.csdn.net/BVL10101111/article/details/72616378](https://blog.csdn.net/BVL10101111/article/details/72616378)
