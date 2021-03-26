# Linear Discriminant Analysis

[Linear Discriminant Analysis](http://en.wikipedia.org/wiki/Linear_discriminant_analysis) (LDA) are statistical analysis methods to find a linear combination of features for separating observations in two classes.
- **Note:** Please refer to [`MulticlassLDA`](@ref) for methods that can discriminate between multiple classes.

## Overview of LDA

Suppose the samples in the positive and negative classes respectively with means: ``\boldsymbol{\mu}_p`` and ``\boldsymbol{\mu}_n``, and covariances ``\mathbf{C}_p`` and ``\mathbf{C}_n``. Then based on *Fisher's Linear Discriminant Criteria*, the optimal projection direction can be expressed as:

```math
    \mathbf{w} = \alpha \cdot (\mathbf{C}_p + \mathbf{C}_n)^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_n)
```
Here ``\alpha`` is an arbitrary non-negative coefficient.

## Linear Discriminant

This package uses the [`LinearDiscriminant`](@ref) type to capture a linear discriminant functional:

```@docs
LinearDiscriminant
```

This type comes with several methods where ``f`` be an instance of  [`LinearDiscriminant`](@ref).

```@docs
fit(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T}; kwargs) where T<:Real
evaluate(::LinearDiscriminant, ::AbstractVector)
evaluate(::LinearDiscriminant, ::AbstractMatrix)
predict(::LinearDiscriminant, ::AbstractVector)
predict(::LinearDiscriminant, ::AbstractMatrix)
coef(::LinearDiscriminant)
dof(::LinearDiscriminant)
weights(::LinearDiscriminant)
length(::LinearDiscriminant)
```

Additional functionality:
```@docs
ldacov
```
