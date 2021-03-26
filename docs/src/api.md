# API Reference

## Current

Table of the package models and corresponding function names used by these models.

| Function \ Model | CCA | WHT | ICA | LDA | FA  |PPCA | PCA |KPCA | MDS |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|fit               |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|transform         |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |  x  |
|predict           |     |     |     |  x  |    |    |    |    |    |
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
|length            |     |     |     |     |    |    |    |    |    |
|size              |     |     |     |     |    |    |    |    |    |

Note: `?` refers to a possible implementation that is missing or called differently.

## New

| Function \ Model | WHT | CCA | LDA |MC-LDA|SS-LDA| ICA | FA  |PPCA | PCA |KPCA | MDS |
|------------------|:---:|:---:|:---:|:----:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
|fit               |  x  |  x  |  x  |  x   |   x  |  x  |  x  |  x  |  x  |  x  |  x  |
|transform         |  x  |  x  |     |  x   |   x  |  x  |  x  |  x  |  x  |  x  |  x  |
|predict           |     |     |  x  |      |      |     |     |     |     |     |     |
|indim             |  -  |     |     |  x   |   x  |  x  |  x  |  x  |  x  |  x  |  x  |
|outdim            |  -  |  x  |     |  x   |   x  |  x  |  x  |  x  |  x  |  x  |  x  |
|mean              |  x  |  x  |     |  x   |   x  |  x  |  x  |  x  |  x  |  ?  |     |
|var               |     |     |     |      |      |     |  x  |  x  |  ?  |  ?  |  ?  |
|cov               |     |     |     |      |      |     |  x  |  ?  |     |     |     |
|cor               |     |  x  |     |      |      |     |     |     |     |     |     |
|projection        |  ?  |  x  |     |      |      |     |  x  |  x  |  x  |  x  |  x  |
|reconstruct       |     |     |     |      |      |     |  x  |  x  |  x  |  x  |     |
|loadings          |     |  ?  |     |      |      |     |  x  |  x  |  ?  |  ?  |  ?  |
|eigvals           |     |     |     |      |      |     |  ?  |  ?  |  ?  |  ?  |  x  |
|eigvecs           |     |     |     |      |      |     | ?   |  ?  |  ?  |  ?  |  ?  |
|length            |  +  |     |  x  |      |      |     |     |     |     |     |     |
|size              |  +  |     |     |      |      |     |     |     |     |     |     |
|                  |     |     |     |      |      |     |     |     |     |     |     |
|eee               |     |     |     |      |      |     |     |     |     |     |     |

- StatsBase.AbstractDataTransform
    - Whitening
      - Interface: fit, transfrom
      - New: length, mean, size
- StatsBase.RegressionModel
    - LinearDiscriminant
      - Methods:
        - Interface: fit, predict, coef, dof, weights
        - New: evaluate, length
    - MulticlassLDA
      - Methods: fit, transfrom, indim, outdim, mean
    - SubspaceLDA
      - Methods: fit, transfrom, indim, outdim, mean
    - CCA
      - Methods: fit, transfrom, indim, outdim, mean
    - Subtypes:
        - AbstractDimensionalityReduction
        - Methods: projection, var, reconstruct, loadings
        - Subtypes:
            - LinearDimensionalityReduction
                - Methods: ICA, PCA
            - NonlinearDimensionalityReduction
                - Methods: KPCA, MDS
            - LatentVariableModel or LatentVariableDimensionalityReduction
                - Methods: FA, PPCA
                - Methods: cov

