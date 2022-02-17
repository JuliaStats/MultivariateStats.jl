# Canonical Correlation Analysis

[Canonical Correlation Analysis](http://en.wikipedia.org/wiki/Canonical_correlation)(CCA) is
a statistical analysis technique to identify correlations between two sets of
variables. Given two vector variables ``X`` and ``Y``, it finds two projections,
one for each, to transform them to a common space with maximum correlations.

The package defines a [`CCA`](@ref) type to represent a CCA model, and provides a set of methods to access the properties.

```@docs
CCA
```

Let `M` be an instance of [`CCA`](@ref), `dx` be the dimension of `X`,
`dy` the dimension of `Y`, and `p` the output dimension (*i.e* the dimension of the common space).

```@docs
fit(::Type{CCA}, ::AbstractMatrix{T}, ::AbstractMatrix{T}) where {T<:Real}
size(::CCA)
mean(::CCA, ::Symbol)
projection(::CCA, ::Symbol)
cor(::CCA)
predict(::CCA, ::AbstractVecOrMat{<:Real}, ::Symbol)
```

Auxiliary functions:

```@docs
ccacov
ccasvd
```

---

### References

[^1]: David Weenink, Canonical Correlation Analysis, Institute of Phonetic Sciences, Univ. of Amsterdam, Proceedings 25, 81-99, 2003.

