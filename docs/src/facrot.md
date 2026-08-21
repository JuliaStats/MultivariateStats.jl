# Factor Rotation

[Factor rotations](https://en.wikipedia.org/wiki/Factor_analysis#Rotation_methods)
are used for post-processing of a factor matrix obtained by
[`FactorAnalysis`](@ref) or [`PCA`](@ref).

Let ``L`` be a matrix of factor loadings. Two common methods of factor rotation[^1]
are typically considered:
- A matrix ``\Lambda`` is an *orthogonal rotation*[^2] of ``L`` if
  ``\Lambda = L R`` for some orthogonal matrix ``R``, i.e. ``R^{-1} = R^\top``.
- A matrix ``\Lambda`` is an *oblique rotation*[^3] of ``L`` if
  ``\Lambda = L (R')^{-1}`` for some matrix ``R`` that has columns of length 1.

The matrix ``R`` is chosen by minimizing a factor rotation criterion
``Q(\Lambda)``. This package uses a gradient projection algorithm[^1]
to minimize the criterion ``Q``. 

## Applying factor rotations

This package defines the function [`rotate`](@ref) which can be used to rotate
a matrix of loadings.

```@docs
rotate(::AbstractMatrix, ::FactorRotationCriterion{T}) where {T<:FactorRotationMethod}
FactorRotation
loadings
rotation
```

[`rotate!`](@ref) modifies the supplied model in place.

```@docs
rotate!(::FactorAnalysis, ::FactorRotationCriterion{T}) where {T<:FactorRotationMethod}
rotate!(::PCA, ::FactorRotationCriterion{T}) where {T<:FactorRotationMethod}
```

## Factor rotation methods

There are two types of factor rotations: *orthogonal* and *oblique*.
Which type of rotation method should be used can be chosen by setting the
appropriate subtype of [`FactorRotationMethod`](@ref).

```@docs
FactorRotationMethod
Orthogonal
Oblique
```

## Factor rotation criteria

The actual factor rotation criteria are described by subtypes of
[`FactorRotationCriterion`](@ref) and many parametrized forms of standard
rotation methods are provided.

```@docs
FactorRotationCriterion{T} where {T<:FactorRotationMethod}
```

Sometimes, factor rotation criteria can be used as orthogonal and oblique methods.

```@docs
CrawfordFerguson{T} where {T<:FactorRotationMethod}
Oblimin{T} where {T<:FactorRotationMethod}
```

### Orthogonal factor rotation criteria

```@docs
Varimax
Quartimax
MinimumEntropy
```

### Oblique factor rotation criteria

```@docs
Quartimin
```

## References

[^1] Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms and Software for Arbitrary Rotation Criteria in Factor Analysis. Educational and Psychological Measurement, 65, 676-696. doi [10.1177/0013164404272507](https://doi.org/10.1177/0013164404272507)
[^2] Jennrich, R. I. (2001). A simple general procedure for orthogonal rotation. Psychometrika, 66, 289-306. doi [10.1007/BF02294840](https://doi.org/10.1007/BF02294840)
[^3] Jennrich, R. I. (2002). A simple general method for oblique rotation. Psychometrika, 67, 7-19. doi [10.1007/BF02294706](https://doi.org/10.1007/BF02294706)

