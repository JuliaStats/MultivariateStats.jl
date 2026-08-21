# API Reference

## Current

Table of the package models and corresponding function names used by these models.

| Function \ Model | CCA | WHT | ICA | LDA | FA  |PPCA | PCA |KPCA | MDS |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|fit               |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|transform         |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|predict           |     |     |     |  x  |     |     |     |     |     |
|indim             |     |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|outdim            |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|mean              |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  ?  |     |
|var               |     |     |     |     |  x  |  x  |  ?  |  ?  |  ?  |
|cov               |     |     |     |     |  x  |  ?  |     |     |     |
|cor               |  x  |     |     |     |     |     |     |     |     |
|projection        |  x  |     |     |     |  x  |  x  |  x  |  x  |  x  |
|reconstruct       |     |     |     |     |  x  |  x  |  x  |  x  |     |
|loadings          |  ?  |     |     |  ?  |  x  |  x  |  ?  |  ?  |  ?  |
|eigvals           |     |     |     |     |  ?  |  ?  |  ?  |  ?  |  x  |
|eigvecs           |     |     |     |     | ?   |  ?  |  ?  |  ?  |  ?  |
|length            |     |     |     |     |     |     |     |     |     |
|size              |     |     |     |     |     |     |     |     |     |

Note: `?` refers to a possible implementation that is missing or called differently.

## New

| Function \ Model | WHT | CCA | LDA |MC-LDA|SS-LDA| ICA | FA  |PPCA | PCA |KPCA | MDS |
|------------------|:---:|:---:|:---:|:----:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
|fit               |  x  |  x  |  x  |  x   |   x  |  x  |  x  |  x  |  x  |  x  |  x  |
|transform         |  x  |  -  |  -  |  -   |   -  |  -  |  x  |  x  |  -  |  -  |  -  |
|predict           |     |  +  |  x  |  +   |   +  |  +  |     |     |  +  |  +  |  +  |
|indim             |  -  |     |     |  -   |   -  |  -  |  x  |  x  |  -  |  -  |  -  |
|outdim            |  -  |  -  |     |  -   |   -  |  -  |  x  |  x  |  -  |  -  |  -  |
|mean              |  x  |  x  |     |  x   |   x  |  x  |  x  |  x  |  x  |     |     |
|var               |     |     |     |      |      |     |  x  |  x  |  x  |     |  ?  |
|cov               |     |     |     |      |      |     |  x  |  x  |     |     |     |
|cor               |     |  +  |     |      |      |     |     |     |     |     |     |
|projection        |  ?  |  x  |     |  x   |   x  |     |  x  |  x  |  x  |  x  |  x  |
|reconstruct       |     |     |     |      |      |     |  x  |  x  |  x  |  x  |     |
|loadings          |     |     |     |      |      |     |  x  |  x  |  x  |     |  +  |
|eigvals           |     |     |     |      |   +  |     |  ?  |  ?  |  x  |  x  |  x  |
|eigvecs           |     |     |     |      |      |     |  ?  |  ?  |  x  |  +  |  +  |
|length            |  +  |     |  +  |  +   |   +  |     |     |     |     |     |     |
|size              |  +  |  +  |     |  +   |   +  |  +  |     |     |  x  |  +  |  +  |
|                  |     |     |     |      |      |     |     |     |     |     |     |

- StatsBase.AbstractDataTransform
    - Whitening
      - Interface: fit, transform
      - New: length, mean, size
- StatsBase.RegressionModel
    - *Interface:* fit, predict
    - LinearDiscriminant
      - Functions: coef, dof, weights, evaluate, length
    - MulticlassLDA
      - Functions: size, mean, projection, length
    - SubspaceLDA
      - Functions: size, mean, projection, length, eigvals
    - CCA
      - Functions: size, mean, projection, predict, cor
    - Subtypes:
        - AbstractDimensionalityReduction
          - *Interface:* projection, var, reconstruct, loadings
          - *Interface:* projection == weights
        - Subtypes:
            - LinearDimensionalityReduction
                - Methods: ICA, PCA
            - NonlinearDimensionalityReduction
                - Methods: KPCA, MDS
                  - Functions: modelmatrix(X),
            - LatentVariableModel or LatentVariableDimensionalityReduction
                - Methods: FA, PPCA
                  - Functions: cov


## Generic Interface

```@meta
CurrentModule = MultivariateStats
```

The models above share a type hierarchy, and the following methods are defined
generically on its abstract types.

```@docs
size(::AbstractDimensionalityReduction, ::Integer)
projection(::AbstractDimensionalityReduction)
reconstruct(::AbstractDimensionalityReduction, ::Any)
loadings(::LinearDimensionalityReduction)
```

## Internals

These are not part of the public API and may change without a breaking release.

```@docs
calcscattermat(::CovarianceEstimator, ::DenseMatrix{T}) where {T<:Real}
toindices(::AbstractVector)
L2distance(::AbstractMatrix{T}) where {T<:Real}
```
