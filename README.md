# AICourse

My take on Andrew Ng's great [Machine Learning](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=1) couse.

This is a project where I'll be developing ML solutions from scratch (no
algebra / ML libraries used) in Rust.
My goal is to learn about how ML works on a deep level.
In addition, I also want to improve my Test-Driven Development and Rust
skillset.

The solutions in this project serve as an example, and are probably not the
most correct or efficient solution to the problem. It could be more easy to
understand than other implementations, given this is written by someone with
little practical experience in machine learning.

# Progress

## Lecture 1-4

There is a Matrix struct that represents a linear algebra matrix.
It has methods for addition, multiplication, inversion amongst other
fundamental operators.

Both gradient descent and normal equation methods are implemented for solving
the multivariate linear regression problem.

The gradient descent implementation "learns" the optimal learning rate by
increasing the learning rate when the new cost is less than the current cost,
and decreasing it when the new cost is higher.
The convergence threshold is currently hardcoded, but precise enough for >99%
accuracy using the datasets in the examples.

The normal equation method can fail on some datasets because singular value
decomposition (SVD) is not implemented for calculating the pseudo-inverse.
