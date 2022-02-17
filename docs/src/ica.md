# Independent Component Analysis

[Independent Component Analysis](http://en.wikipedia.org/wiki/Independent_component_analysis) (ICA) is
a computational technique for separating a multivariate signal into additive subcomponents,
with the assumption that the subcomponents are non-Gaussian and independent from each other.

There are multiple algorithms for ICA. Currently, this package implements the Fast ICA algorithm.

## FastICA

This package implements the FastICA algorithm[^1]. The package uses the [`ICA`](@ref) type to define a FastICA model:

```@docs
ICA
```

Several methods are provided to work with [`ICA`](@ref). Let ``M`` be an instance of [`ICA`](@ref):

```@docs
fit(::Type{ICA}, ::AbstractMatrix{T}, ::Int) where {T<:Real}
size(::ICA)
mean(::ICA)
predict(::ICA, ::AbstractVecOrMat{<:Real})
```

The package also exports functions of the core algorithms. Sometimes, it can be more
efficient to directly invoke them instead of going through the `fit` interface.

```@docs
fastica!
```

The FastICA method requires a first derivative of a functor ``g`` to approximate negative entropy.
The package implements an following interface for defining derivative value estimation:

```@docs
MultivariateStats.ICAGDeriv
MultivariateStats.Tanh
MultivariateStats.Gaus
```

---

### References

[^1]: Aapo Hyvarinen and Erkki Oja, Independent Component Analysis: Algorithms and Applications. Neural Network 13(4-5), 2000.
