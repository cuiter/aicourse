# AICourse

My take on Andrew Ng's great [Machine Learning](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=1) couse.

I'll be developing ML solutions from scratch (no external algebra / ML
libraries required) in Rust.
The goal is to learn about how ML works on a deep level.

The solutions in this project serve as an example, and are probably not the
most correct or efficient solution to the problem. Perhaps it is more readable
than other implementations, given this is written by someone who has recently
started learning about machine learning.

To see a working neural network in action, clone the project including
submodules and run:
`cargo run --release`

This will train a network to classify handwritten digits using the MNIST
dataset.

# Progress

## Lecture 1-4

There is a linear algebra Matrix implementation with methods for addition,
multiplication, inversion amongst other fundamental operators.

The multivariate linear regression problem can be solved with both gradient
descent and normal equation methods.

The gradient descent implementation "learns" the optimal learning rate by
increasing the learning rate when the new cost is less than the current cost,
and decreasing it when the new cost is higher.
The convergence threshold is currently hardcoded, but precise enough for
the datasets in the examples.

The normal equation method can fail on some datasets because singular value
decomposition (SVD) is not implemented for calculating the pseudo-inverse of a
matrix.

Polynomial regression is implemented by letting the user pass a transformation
function which transforms the input into the inputs used by the learning
algorithm.

## Lecture 5-7

Logistic regression in both single-classification and one-vs-all are
implemented. Regularization is implemented on all methods, though I am not
certain whether the logistic regression regularization methods are correct.

## Lecture 8-9

A deep feed-forward neural network using the logistic sigmoid activation
function is implemented. It's not clear whether regularization works
correctly, but the behaviour seems to match the lectures.

The network can be trained with different regularization
parameters in parallel. Proper splitting up of train / cross-validation / test
datasets is not implemented yet, so the results can be skewed.

After training a [28 * 28, 256, 10] unit network in parallel for
~70 epochs with the first 5000 samples of the MNIST train dataset, it is able
to classify the MNIST test dataset with an accuracy of ~92%. This takes
approximately 7 minutes with a Ryzen 5 1600 CPU.
