# Regression

The package provides functions to perform *Linear Least Square*, *Ridge*, and *Isotonic Regression*.

## Examples

```@setup REGex
using Plots
gr(fmt=:svg)
```

Performing [`llsq`](@ref) regression on *cars* data set:

```@example REGex
using MultivariateStats, RDatasets, Plots

# load cars dataset
cars = dataset("datasets", "cars")

# calculate regression models
a = llsq(cars[!,:Speed], cars[!, :Dist])
b = isotonic(cars[!,:Speed], cars[!, :Dist])

# plot results
p = scatter(cars[!,:Speed], cars[!,:Dist], xlab="Speed", ylab="Distance",
            leg=:topleft, lab="data")
let xs = cars[!,:Speed]
    plot!(p, xs, map(x->a[1]*x+a[2], xs), lab="llsq")
    plot!(p, xs, b, lab="isotonic")
end
```

For a single response vector `y` (without using bias):

```julia
# prepare data
X = rand(1000, 3)               # feature matrix
a0 = rand(3)                    # ground truths
y = X * a0 + 0.1 * randn(1000)  # generate response

# solve using llsq
a = llsq(X, y; bias=false)

# do prediction
yp = X * a

# measure the error
rmse = sqrt(mean(abs2.(y - yp)))
print("rmse = $rmse")
```

For a single response vector `y` (using bias):

```julia
# prepare data
X = rand(1000, 3)
a0, b0 = rand(3), rand()
y = X * a0 .+ b0 .+ 0.1 * randn(1000)

# solve using llsq
sol = llsq(X, y)

# extract results
a, b = sol[1:end-1], sol[end]

# do prediction
yp = X * a .+ b'
```

For a matrix of column-stored regressors `X` and a matrix comprised of multiple columns of dependent variables `Y`:

```julia
# prepare data
X = rand(3, 1000)
A0, b0 = rand(3, 5), rand(1, 5)
Y = (X' * A0 .+ b0) + 0.1 * randn(1000, 5)

# solve using llsq
sol = llsq(X, Y, dims=2)

# extract results
A, b = sol[1:end-1,:], sol[end,:]

# do prediction
Yp = X'*A .+ b'
```

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

The package provides following functions to solve the above problems:

```@docs
llsq
```

## Ridge Regression

Compared to linear least square, [Ridge Regression](http://en.wikipedia.org/wiki/Tikhonov_regularization>)
uses an additional quadratic term to regularize the problem:

```math
\mathop{\mathrm{minimize}}_{(\mathbf{a}, b)} \
    \frac{1}{2} \|\mathbf{y} - (\mathbf{X} \mathbf{a} + b)\|^2 +
    \frac{1}{2} \mathbf{a}^T \mathbf{Q} \mathbf{a}
```

The transposed form:

```math
    \mathop{\mathrm{minimize}}_{(\mathbf{a}, b)} \
    \frac{1}{2} \|\mathbf{y} - (\mathbf{X}^T \mathbf{a} + b)\|^2 +
    \frac{1}{2} \mathbf{a}^T \mathbf{Q} \mathbf{a}
```

The package provides following functions to solve the above problems:

```@docs
ridge
```

## Isotonic Regression

[Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression) or
monotonic regression fits a sequence of observations into a fitted line that is
non-decreasing (or non-increasing) everywhere. The problem defined as a weighted
least-squares fit ``{\hat {y}}_{i} \approx y_{i}`` for all ``i``, subject to
the constraint that ``{\hat {y}}_{i} \leq {\hat {y}}_{j}`` whenever
``x_{i} \leq x_{j}``.
This gives the following quadratic program:

```math
\min \sum_{i=1}^{n} w_{i}({\hat {y}}_{i}-y_{i})^{2}
\text{  subject to  } {\hat {y}}_{i} \leq {\hat {y}}_{j}
\text{ for all } (i,j) \in E
```
where ``E=\{(i,j):x_{i}\leq x_{j}\}`` specifies the partial ordering of
the observed inputs ``x_{i}``.

The package provides following functions to solve the above problems:
```@docs
isotonic
```

---

### References

[^1]: Best, M.J., Chakravarti, N. Active set algorithms for isotonic regression; A unifying framework. Mathematical Programming 47, 425–439 (1990).

