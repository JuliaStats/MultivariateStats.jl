# Principal Component Analysis

[Principal Component Analysis](http://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) derives an orthogonal projection to convert a given set of observations to linearly uncorrelated variables, called *principal components*.

## Example

Performing [`PCA`](@ref) on *Iris* data set:

```@example PCAex
using MultivariateStats, RDatasets, Plots
plotly() # using plotly for 3D-interacive graphing

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

Suppose `Xtr` and `Xte` are training and testing data matrix, with each observation in a column.
We train a PCA model, allowing up to 3 dimensions:

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

Now, we group results by testing set labels for color coding and visualize first 3 principal
components in 3D interactive plot

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
mean(M::PCA)
projection(M::PCA)
var(M::PCA)
tprincipalvar(M::PCA)
tresidualvar(M::PCA)
r2(M::PCA)
loadings(M::PCA)
eigvals(M::PCA)
eigvecs(M::PCA)
```

Auxiliary functions
```@docs
pcacov
pcasvd
```
