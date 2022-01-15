# Regression

The package provides functions to perform *Linear Least Square*, *Ridge*, and *Isotonic Regression*.


## Linear Least Square

[Linear Least Square](http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics))
is to find linear combination(s) of given variables to fit the responses by
minimizing the squared error between them.
This can be formulated as an optimization as follows:

```math
\mathop{\mathrm{minimize}}_{(\mathbf{a}, b)} \
    \frac{1}{2} \|\mathbf{y} - (\mathbf{X} \mathbf{a} + b)\|^2
```

Sometimes, the coefficient matrix is given in a transposed form, in which case,
the optimization is modified as:

```math
\mathop{\mathrm{minimize}}_{(\mathbf{a}, b)} \
    \frac{1}{2} \|\mathbf{y} - (\mathbf{X}^T \mathbf{a} + b)\|^2
```

