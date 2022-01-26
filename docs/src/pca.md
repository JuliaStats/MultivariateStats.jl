# Principal Component Analysis

[Principal Component Analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) derives an orthogonal projection to convert a given set of observations to linearly uncorrelated variables, called *principal components*.

## Example

```@setup PCAex
using Plots
gr(fmt=:svg)
```

Performing [`PCA`](@ref) on *Iris* data set:

```@example PCAex
using MultivariateStats, RDatasets, Plots

# load iris dataset
iris = dataset("datasets", "iris")

# split half to training set
Xtr = Matrix(iris[1:2:end,1:4])'
Xtr_labels = Vector(iris[1:2:end,5])

# split other half to testing set
Xte = Matrix(iris[2:2:end,1:4])'
Xte_labels = Vector(iris[2:2:end,5])
nothing # hide
```

Suppose `Xtr` and `Xte` are training and testing data matrix, with each observation
in a column. We train a PCA model, allowing up to 3 dimensions:

```@example PCAex
M = fit(PCA, Xtr; maxoutdim=3)
```

Then, apply PCA model to the testing set

```@example PCAex
Yte = predict(M, Xte)
```

And, reconstruct testing observations (approximately) to the original space

```@example PCAex
Xr = reconstruct(M, Yte)
```

Now, we group results by testing set labels for color coding and visualize first
3 principal components in 3D plot

```@example PCAex
setosa = Yte[:,Xte_labels.=="setosa"]
versicolor = Yte[:,Xte_labels.=="versicolor"]
virginica = Yte[:,Xte_labels.=="virginica"]

p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
plot!(p,xlabel="PC1",ylabel="PC2",zlabel="PC3")
```

## Linear Principal Component Analysis

This package uses the [`PCA`](@ref) type to define a linear PCA model:

```@docs
PCA
```

This type comes with several methods where ``M`` be an instance of  [`PCA`](@ref),
``d`` be the dimension of observations, and ``p`` be the output dimension (*i.e* the dimension of the principal subspace).

```@docs
fit(::Type{PCA}, ::AbstractMatrix{T}; kwargs) where {T<:Real}
predict(::PCA, ::AbstractVecOrMat{T}) where {T<:Real}
reconstruct(::PCA, ::AbstractVecOrMat{T}) where {T<:Real}
size(::PCA)
mean(::PCA)
projection(::PCA)
var(::PCA)
principalvars(::PCA)
tprincipalvar(::PCA)
tresidualvar(::PCA)
r2(::PCA)
loadings(::PCA)
eigvals(::PCA)
eigvecs(::PCA)
```

Auxiliary functions:
```@docs
pcacov
pcasvd
```

## Kernel Principal Component Analysis

[Kernel Principal Component Analysis](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis>) (kernel PCA)
is an extension of principal component analysis (PCA) using techniques of kernel methods.
Using a kernel, the originally linear operations of PCA are performed in a reproducing kernel Hilbert space.


This package defines a [`KernelPCA`](@ref) type to represent a kernel PCA model.

```@docs
KernelPCA
```

The package provides a set of methods to access the properties of the kernel PCA
model. Let ``M`` be an instance of [`KernelPCA`](@ref), ``d`` be the dimension of
observations, and ``p`` be the output dimension (*i.e* the dimension of the principal subspace).

```@docs
fit(::Type{KernelPCA}, ::AbstractMatrix{T}; kwargs...) where {T<:Real}
predict(::KernelPCA)
predict(::KernelPCA, ::AbstractVecOrMat{T}) where {T<:Real}
reconstruct(::KernelPCA, ::AbstractVecOrMat{T}) where {T<:Real}
size(::KernelPCA)
projection(::KernelPCA)
eigvals(::KernelPCA)
eigvecs(::KernelPCA)
```

### Kernels

List of the commonly used kernels:

| function | description |
|----------|-------------|
|`(x,y)->x'y`| Linear |
|`(x,y)->(x'y+c)^d`| Polynomial |
|`(x,y)->exp(-γ*norm(x-y)^2.0)`| Radial basis function (RBF) |

This package has a separate interface for adjusting kernel matrices.

```@docs
MultivariateStats.KernelCenter
fit(::Type{MultivariateStats.KernelCenter}, ::AbstractMatrix{<:Real})
MultivariateStats.transform!(::MultivariateStats.KernelCenter, ::AbstractMatrix{<:Real})
```


## Probabilistic Principal Component Analysis

[Probabilistic Principal Component Analysis](https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis) (PPCA)
represents a constrained form of the Gaussian distribution in which the number of
free parameters can be restricted while still allowing the model to capture
the dominant correlations in a data set. It is expressed as the maximum likelihood
solution of a probabilistic latent variable model[^1].

This package defines a [`PPCA`](@ref) type to represent a probabilistic PCA model,
and provides a set of methods to access the properties.

```@docs
PPCA
```

Let ``M`` be an instance of [`PPCA`](@ref), ``d`` be the dimension of observations,
and ``p`` be the output dimension (*i.e* the dimension of the principal subspace).

```@docs
fit
size(::PPCA)
mean(::PPCA)
var(::PPCA)
cov(::PPCA)
projection(::PPCA)
loadings(::PPCA)
```

Given a probabilistic PCA model ``M``, one can use it to transform observations into
latent variables, as

```@math
\mathbf{z} = (\mathbf{W}^T \mathbf{W} + \sigma^2 \mathbf{I}) \mathbf{W}^T (\mathbf{x} - \boldsymbol{\mu})
```

or use it to reconstruct (approximately) the observations from latent variables, as

```@math
\tilde{\mathbf{x}} = \mathbf{W} \mathbb{E}[\mathbf{z}] + \boldsymbol{\mu}
```

Here, ``\mathbf{W}`` is the factor loadings or weight matrix.

```@docs
predict(::PPCA, ::AbstractVecOrMat{T}) where {T<:Real}
reconstruct(::PPCA, ::AbstractVecOrMat{T}) where {T<:Real}
```

Auxiliary functions:

```@docs
ppcaml
ppcaem
bayespca
```

---

### References

[^1]: Bishop, C. M. Pattern Recognition and Machine Learning, 2006.
