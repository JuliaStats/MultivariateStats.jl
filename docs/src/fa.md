# Factor Analysis

[Factor Analysis](https://en.wikipedia.org/wiki/Factor_analysis) (FA) is
a linear-Gaussian latent variable model that is closely related to probabilistic PCA.
In contrast to the probabilistic PCA model, the covariance of conditional distribution of
the observed variable  given the latent variable is diagonal rather than isotropic[^1].

This package defines a [`FactorAnalysis`](@ref) type to represent a factor analysis
model, and provides a set of methods to access the properties.

```@docs
FactorAnalysis
```

The package provides a set of methods to access the properties of the factor analysis
model. Let ``M`` be an instance of [`FactorAnalysis`](@ref), ``d`` be the dimension of
observations, and ``p`` be the output dimension (*i.e* the dimension of the principal
subspace).

```@docs
fit(::Type{FactorAnalysis}, ::AbstractVecOrMat{<:Real})
size(::FactorAnalysis)
mean(::FactorAnalysis)
var(::FactorAnalysis)
cov(::FactorAnalysis)
projection(::FactorAnalysis)
loadings(::FactorAnalysis)
```

Given a factor analysis model ``M``, one can use it to transform observations into
latent variables, as

```@math
\mathbf{z} =  \mathbf{W}^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
```

or use it to reconstruct (approximately) the observations from latent variables, as

```@math
\tilde{\mathbf{x}} = \mathbf{\Sigma} \mathbf{W} (\mathbf{W}^T \mathbf{W})^{-1} \mathbf{z} + \boldsymbol{\mu}
```

Here, ``\mathbf{W}`` is the factor loadings or weight matrix,
``\mathbf{\Sigma} = \mathbf{\Psi} + \mathbf{W}^T \mathbf{W}`` is the covariance matrix.

The package provides methods to do so:

```@docs
predict(::FactorAnalysis, ::AbstractVecOrMat{<:Real})
reconstruct(::FactorAnalysis, ::AbstractVecOrMat{<:Real})
```

Auxiliary functions:

```@docs
faem
facm
```

---

### References

[^1]: Bishop, C. M. Pattern Recognition and Machine Learning, 2006.
[^2]:  Rubin, Donald B., and Dorothy T. Thayer. EM algorithms for ML factor analysis. Psychometrika 47.1, 69-76, 1982.
[^3]:  Zhao, J-H., Philip LH Yu, and Qibao Jiang. ML estimation for factor analysis: EM or non-EM?. Statistics and computing 18.2, 109-123, 2008.

