# Data Transformation

## Whitening

A [whitening transformation](http://en.wikipedia.org/wiki/Whitening_transformation>) is a decorrelation transformation that transforms a set of random variables into a set of new random variables with identity covariance (uncorrelated with unit variances).

In particular, suppose a random vector has covariance ``\mathbf{C}``, then a whitening transform ``\mathbf{W}`` is one that satisfy:

```math
   \mathbf{W}^T \mathbf{C} \mathbf{W} = \mathbf{I}
```

Note that ``\mathbf{W}`` is generally not unique. In particular, if ``\mathbf{W}`` is a whitening transform, so is any of its rotation ``\mathbf{W} \mathbf{R}`` with ``\mathbf{R}^T \mathbf{R} = \mathbf{I}``.

The package uses [`Whitening`](@ref) to represent a whitening transform.

```@docs
Whitening
```

Whitening transformation can be fitted to data using the `fit` method.

```@docs
fit(::Type{Whitening}, X::AbstractMatrix{T}; kwargs...) where {T<:Real}
transform(::Whitening, ::AbstractVecOrMat)
length(::Whitening)
mean(::Whitening)
size(::Whitening)
```

Additional methods
```@docs
cov_whitening
cov_whitening!
```
