"""
Metric MDS
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

function stress(D::AbstractMatrix{T}, Δ::AbstractMatrix{T},
                W::AbstractMatrix{T} = ones(size(Δ));
                η_δ::Real = NaN) where {T<:Real}
    n, m = size(D)
    @assert n == m "Matrix must be symmetric"
    @assert (n,m) == size(Δ) "Matrix size mismatch"
    η_δ = isnan(η_δ) ? sum(Δ.^2)/2 : η_δ # unweighted sq. dissimilarities

    η = ρ = zero(T)
    @inbounds for j in 1:n, i in (j+1):n
        d = D[j, i]
        w = W[j,i]
        η += w*d*d
        ρ += w*Δ[j,i]*d
    end

    return η_δ + η - 2ρ
end

isidentity(f::typeof(identity)) = true
isidentity(f) = false
e
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
    - `identity`: use dissimilarity values as the disparities to perform the metric MDS (*default*)
    - `isotonic`: converts dissimilarity values to ordinal disparities to perform non-metric MDS
    - any univariate disparity transformation function
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
             metric::Function = identity,
             tol::Real = 1e-3,
             maxiter::Int = 300,
             initial::Union{Nothing,AbstractMatrix{T}} = nothing,
             weights::Union{Nothing,AbstractMatrix{T}} = nothing,
             distances::Bool) where {T<:Real}

    # get distance matrix and space dimension
    Δ, d = if !distances
        L2distance(X), size(X,1)
    else
        X, NaN
    end
    n = size(X,2)
    ismetric = isidentity(metric)

    W = weights === nothing ? ones(size(Δ)) : weights
    η_δ = ismetric ? sum(abs2, W.*Δ)/2 : n*(n-1)/2

    # reduce space coordinates
    Z = initial === nothing ? rand(T, maxoutdim, n) : copy(initial)
    Y = copy(Z)
    D = L2distance(Z)

    i = 1
    converged = false
    σ′= Inf
    chg = NaN
    while i <= maxiter

        # Guttman transform
        D[iszero.(D)] .= eps(T)
        B = if weights === nothing
            -Δ./D
        else
            -W.*Δ./D
        end
        B[diagind(B)] .= -sum(B, dims=2)
        mul!(Y, Z, B)
        @. Z = Y/n

        # calculate distances in reduced space
        L2distance!(D, Z)

        # calculate stress
        σ = stress(D, Δ, W; η_δ=η_δ)

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

