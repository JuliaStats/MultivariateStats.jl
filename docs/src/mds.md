# Multidimensional Scaling

In general, [Multidimensional Scaling](http://en.wikipedia.org/wiki/Multidimensional_scaling) (MDS)
refers to techniques that transforms samples into lower dimensional space while
preserving the inter-sample distances as well as possible.

## Example

```@setup MDSex
using Plots
gr(fmt=:svg)
```

Performing [`MDS`](@ref) on *Iris* data set:

```@example MDSex
using MultivariateStats, RDatasets, Plots

# load iris dataset
iris = dataset("datasets", "iris")

# take half of the dataset
X = Matrix(iris[1:2:end,1:4])'
X_labels = Vector(iris[1:2:end,5])
nothing # hide
```

Suppose `X` is our data matrix, with each observation in a column.
We train a MDS model, allowing up to 3 dimensions:

```@example MDSex
M = fit(MDS, X; maxoutdim=3, distances=false)
```

Then, apply MDS model to get an embedding of our data in 3D space:

```@example MDSex
Y = predict(M)
```

Now, we group results by testing set labels for color coding and visualize first
3 principal components in 3D interactive plot

```@example MDSex
setosa = Y[:,X_labels.=="setosa"]
versicolor = Y[:,X_labels.=="versicolor"]
virginica = Y[:,X_labels.=="virginica"]

p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
```

## Classical Multidimensional Scaling
This package defines a `MDS` type to represent a classical MDS model [^1],
and provides a set of methods to access the properties.

```@docs
MDS
```

The MDS method type comes with several methods where ``M`` be an instance of [`MDS`](@ref),
``d`` be the dimension of observations, and ``p`` be the output dimension, i.e.
the embedding dimension, and ``n`` is the number of the observations.

```@docs
fit(::Type{MDS}, ::AbstractMatrix{T}; kwargs) where {T<:Real}
predict(::MDS)
predict(::MDS, ::AbstractVector{<:Real})
size(::MDS)
projection(M::MDS)
loadings(M::MDS)
eigvals(M::MDS)
eigvecs(M::MDS)
stress
```

This package provides following functions related to classical MDS.
```@docs
gram2dmat
gram2dmat!
dmat2gram
dmat2gram!
```

## References

[^1]: Ingwer Borg and Patrick J. F. Groenen, "Modern Multidimensional Scaling: Theory and Applications", Springer, pp. 201–268, 2005.

