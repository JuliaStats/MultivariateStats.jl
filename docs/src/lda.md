# Linear Discriminant Analysis

[Linear Discriminant Analysis](http://en.wikipedia.org/wiki/Linear_discriminant_analysis) (LDA) are statistical analysis methods to find a linear combination of features for separating observations in two classes.
- **Note:** Please refer to [`MulticlassLDA`](@ref) for methods that can discriminate between multiple classes.

Suppose the samples in the positive and negative classes respectively with means: ``\boldsymbol{\mu}_p`` and ``\boldsymbol{\mu}_n``, and covariances ``\mathbf{C}_p`` and ``\mathbf{C}_n``. Then based on *Fisher's Linear Discriminant Criteria*, the optimal projection direction can be expressed as:

```math
\mathbf{w} = \alpha \cdot (\mathbf{C}_p + \mathbf{C}_n)^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_n)
```
Here ``\alpha`` is an arbitrary non-negative coefficient.

## Two-class Linear Discriminant Analysis

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

## Multi-class Linear Discriminant Analysis

*Multi-class LDA* is a generalization of standard two-class LDA that can handle arbitrary number of classes.

### Overview

*Multi-class LDA* is based on the analysis of two scatter matrices: *within-class scatter matrix* and *between-class scatter matrix*.

Given a set of samples ``\mathbf{x}_1, \ldots, \mathbf{x}_n``, and their class labels ``y_1, \ldots, y_n``:

- The **within-class scatter matrix** is defined as:
```math
\mathbf{S}_w = \sum_{i=1}^n (\mathbf{x}_i - \boldsymbol{\mu}_{y_i}) (\mathbf{x}_i - \boldsymbol{\mu}_{y_i})^T
```
Here, ``\boldsymbol{\mu}_k`` is the sample mean of the ``k``-th class.

- The **between-class scatter matrix** is defined as:
```math
\mathbf{S}_b = \sum_{k=1}^m n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) (\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
```
Here, ``m`` is the number of classes, ``\boldsymbol{\mu}`` is the overall sample mean, and ``n_k`` is the number of samples in the ``k``-th class.

Then, multi-class LDA can be formulated as an optimization problem to find a set of linear combinations (with coefficients ``\mathbf{w}``) that maximizes the ratio of the between-class scattering to the within-class scattering, as

```math
\hat{\mathbf{w}} = \mathop{\mathrm{argmax}}_{\mathbf{w}}
    \frac{\mathbf{w}^T \mathbf{S}_b \mathbf{w}}{\mathbf{w}^T \mathbf{S}_w \mathbf{w}}
```

The solution is given by the following generalized eigenvalue problem:

```math
\mathbf{S}_b \mathbf{w} = \lambda \mathbf{S}_w \mathbf{w}
```

Generally, at most ``m - 1`` generalized eigenvectors are useful to discriminate between ``m`` classes.

When the dimensionality is high, it may not be feasible to construct the scatter matrices explicitly. In such cases, see [`SubspaceLDA`](@ref) below.

### Normalization by number of observations

An alternative definition of the within- and between-class scatter matrices normalizes for the number of observations in each group:

```math
\mathbf{S}_w^* = n \sum_{k=1}^m \frac{1}{n_k} \sum_{i \mid y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_{k}) (\mathbf{x}_i - \boldsymbol{\mu}_{k})^T
```

```math
\mathbf{S}_b^* = n \sum_{k=1}^m (\boldsymbol{\mu}_k - \boldsymbol{\mu}^*) (\boldsymbol{\mu}_k - \boldsymbol{\mu}^*)^T
```

where
```math
\boldsymbol{\mu}^* = \frac{1}{k} \sum_{k=1}^m \boldsymbol{\mu}_k.
```

This definition can sometimes be more useful when looking for directions which discriminate among clusters containing widely-varying numbers of observations.

### Multi-class LDA

The package defines a [`MulticlassLDA`](@ref) type to represent a multi-class LDA model, as:

```@docs
MulticlassLDA
MulticlassLDAStats
```

Several methods are provided to access properties of the LDA model. Let `M` be an instance of `MulticlassLDA`:

```@docs
fit(::Type{MulticlassLDA}, ::Int, ::DenseMatrix{T}, ::AbstractVector{Int}; kwargs...) where T<:Real
predict(::MulticlassLDA, ::AbstractVecOrMat{T}) where {T<:Real}
mean(::MulticlassLDA)
size(::MulticlassLDA)
length(::MulticlassLDA)
classmeans(::MulticlassLDA)
classweights(::MulticlassLDA)
withclass_scatter(::MulticlassLDA)
betweenclass_scatter(::MulticlassLDA)
projection(::MulticlassLDA)
```

## Subspace Linear Discriminant Analysis

The package also defines a [`SubspaceLDA`](@ref) type to represent a multi-class LDA model for high-dimensional spaces.
[`MulticlassLDA`](@ref), because it stores the scatter matrices, is not well-suited for high-dimensional data.
For example, if you are performing LDA on images, and each image has ``10^6`` pixels, then the scatter matrices
would contain ``10^12`` elements, far too many to store directly.
[`SubspaceLDA`](@ref) calculates the projection direction without the intermediary of the scatter matrices,
by focusing on the subspace that lies within the span of the within-class scatter. This also
serves to regularize the computation.

[`SubspaceLDA`](@ref) supports all the same methods as [`MulticlassLDA`](@ref), with the exception of
the functions that return a scatter matrix.  The overall projection is represented as
a factorization ``P*L``, where ``P'*x`` projects data points to the subspace spanned by the within-class
scatter, and ``L`` is the LDA projection in the subspace.  The projection directions ``w``
(the columns of ``projection(M)``) satisfy the equation

```math
   \mathbf{P}^T \mathbf{S}_b \mathbf{w} = \lambda \mathbf{P}^T \mathbf{S}_w \mathbf{w}.
```

When ``P`` is of full rank (e.g., if there are more data points than dimensions), then this equation guarantees that
``\mathbf{S}_b \mathbf{w} = \lambda \mathbf{S}_w \mathbf{w}`` will also hold.

The package defines a [`SubspaceLDA`](@ref) type to represent a multi-class LDA model, as:

```@docs
SubspaceLDA
```

Several methods are provided to access properties of the LDA model. Let `M` be an instance of [`SubspaceLDA`](@ref):

```@docs
fit(::Type{SubspaceLDA}, ::DenseMatrix{T}, ::AbstractVector{Int}, ::Int; kwargs...) where T<:Real
predict(::SubspaceLDA, ::AbstractVecOrMat{T}) where {T<:Real}
mean(::SubspaceLDA)
projection(::SubspaceLDA)
size(::SubspaceLDA)
length(::SubspaceLDA)
eigvals(::SubspaceLDA)
```
