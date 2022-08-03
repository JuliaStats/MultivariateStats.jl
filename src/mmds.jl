"""
(Non)Metric Multidimensional Scaling

Use this class to perform (non)metric multidimensional scaling.

There are two calling options, specified via the required keyword argument `distances`:

```julia
mds = fit(MDS, X; distances=false, maxoutdim=size(X,1)-1)
```

where `X` is the data matrix. Distances between pairs of columns of `X` are computed using the Euclidean norm.
This is equivalent to performing PCA on `X`.

```julia
mds = fit(MDS, D; distances=true, maxoutdim=size(D,1)-1)
```

where `D` is a symmetric matrix `D` of distances between points.

In addition, the `metric` parameter specifies type of MDS, By default, it is assigned with `nothing` value
which results in performing *metric MDS* with dissimilarities calculated as Euclidian distances.

An arbitrary transformation function can be provided to `metric` parameter to
perform metric MDS with transformed proximities. The function has to accept two parameters,
a vector of proximities and a vector of distances, corresponding to the proximities, to calculate
disparities required for stress calculation. Internally, the proximity and distance vectors are
obtained from compact triangular matrix representation of proximity and distance matrices.

For *ratio MDS*, a following ratio transformation function can be used

```julia
mds = fit(MDS, D; distances=true, metric=((p,d)->2 .* p))
```

For *order MDS*, use `isotonic` regression function in the `metric` parameter:

```julia
mds = fit(MDS, D; distances=true, metric=isotonic)
```
"""
struct MetricMDS{T<:Real} <: NonlinearDimensionalityReduction
    d::Real         # original dimension
    X::Matrix{T}    # fitted data (n points in m-dim space: m x n)
    σ::T            # stress
end

function show(io::IO, M::MetricMDS)
    d, p = size(M)
    print("Metric MDS(indim = $d, outdim = $p)")
end

## properties
"""
    size(M::MetricMDS)

Returns tuple where the first value is the MDS model `M` input dimension,
*i.e* the dimension of the observation space, and the second value is the output
dimension, *i.e* the dimension of the embedding.
"""
size(M::MetricMDS) = (M.d, size(M.X,1))

"""
    predict(M::MetricMDS)

Returns a coordinate matrix of size ``(p, n)`` for the MDS model `M`, where each column
is the coordinates for an observation in the embedding space.
"""
predict(M::MetricMDS) = M.X

"""
    stress(M::MetricMDS)

Get the stress of the MDS model `M`.
"""
stress(M::MetricMDS) = M.σ

function stress(D::AbstractMatrix{T}, Δ::AbstractMatrix{<:Real},
                W::AbstractMatrix{T} = ones(T, size(Δ));
                η_δ::Real = NaN) where {T<:Real}
    n, m = size(D)
    @assert n == m "Matrix must be symmetric"
    @assert (n,m) == size(Δ) "Matrix size mismatch"
    η_δ = isnan(η_δ) ? sum(abs2, W.*Δ)/2 : η_δ

    η = ρ = zero(T)
    @inbounds for j in 1:n, i in (j+1):n
        d = D[j, i]
        w = W[j,i]
        η += w*d*d
        ρ += w*Δ[j,i]*d
    end

    return η_δ + η - 2ρ
end

"""
    fit(MetricMDS, X; kwargs...)

Compute an embedding of `X` points by (non)metric multidimensional scaling (MDS).

**Keyword arguments:**

Let `(d, n) = size(X)` be respectively the input dimension and the number of observations:

- `distances`: The choice of input (*required*):
    - `false`: use `X` to calculate dissimilarity matrix using Euclidean distance
    - `true`: use `X` input as precomputed dissimilarity symmetric matrix (distances)
- `maxoutdim`: Maximum output dimension (*default* `d-1`)
- `metric` : a function for calculation of disparity values
    - `nothing`: use dissimilarity values as the disparities to perform the metric MDS (*default*)
    - `isotonic`: converts dissimilarity values to ordinal disparities to perform non-metric MDS
    - any two parameter disparity transformation function, where the first parameter is a vector of proximities (i.e. dissimilarities) and the second parameter is a vector of distances, e.g. `(p,d)->b*p` for some `b` is a transformation function for *ratio* MDS.
- `tol`: Convergence tolerance (*default* `1.0e-3`)
- `maxiter`: Maximum number of iterations (*default* `300`)
- `initial`: an initial reduced space point configuration
    - `nothing`: then an initial configuration is randomly generated (*default*)
    - pre-defined matrix
- `weights`: a weight matrix
    - `nothing`: then weights are set to one, ``w_{ij} = 1`` (*default*)
    - pre-defined matrix

*Note:* if the algorithm is unable to converge then `ConvergenceException` is thrown.
"""
function fit(::Type{MetricMDS}, X::AbstractMatrix{T};
             maxoutdim::Int = size(X,1)-1,
             metric::Union{Nothing,Function} = nothing,
             tol::Real = 1e-3,
             maxiter::Int = 300,
             initial::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
             weights::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
             distances::Bool) where {T<:Real}

    # get distance matrix and space dimension
    Δ, d = if !distances
        L2distance(X), size(X,1)
    else
        X, NaN
    end
    n = size(X,2)
    ismetric = metric === nothing
    T2 = promote_type(T, Float32)

    # initialize weights
    W = weights === nothing ? ones(T2, size(Δ)) : weights

    # reduce space coordinates
    Z = initial === nothing ? rand(T2, maxoutdim, n) : copy(initial)

    # temporary collections of matrix elements
    D = zeros(T2, n, n)
    Y = copy(Z)
    Dhat = collect(D)
    utidx = findall(UpperTriangular(Δ) .> 0)
    vΔ = view(Δ, utidx)
    vD = view(D, utidx)
    vDhat = view(Dhat, utidx)
    didx = diagind(D)
    diagD = view(D, didx)

    if !ismetric
        η_δ = NaN
    else
        Dhat = Δ
        η_δ = sum(abs2, W.*Δ)/2
    end

    i = 1
    converged = false
    σ′= Inf
    chg = NaN
    while i <= maxiter
        # calculate distances in reduced space
        L2distance!(D, Z)

        # calculate disparities
        if !ismetric
            disparities = metric(vΔ, vD)
            vDhat .= disparities
            LinearAlgebra.copytri!(Dhat, 'U')
            Dhat .*= sqrt(n*(n-1)/2/sum(abs2, Dhat))
        end

        # calculate stress
        σ = stress(D, Dhat, W, η_δ=η_δ)

        # Guttman transform
        diagD .= eps(T2)
        B = if weights === nothing
            -Dhat./D
        else
            -W.*Dhat./D
        end
        B[didx] .= -sum(B, dims=2) |> vec
        mul!(Y, Z, B)
        @. Z = Y/n

        chg = abs(σ′ - σ)
        @debug "Stress" iter=i σ=σ Δσ=chg
        if chg < tol
            converged = true
            break
        end
        σ′ = σ
        i += 1
    end
    converged || throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))

    MetricMDS(d, Z, σ′)
end

