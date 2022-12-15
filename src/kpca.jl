# Kernel Principal Component Analysis

using Arpack: eigs

"""Center a kernel matrix"""
struct KernelCenter{T<:Real}
    means::AbstractVector{T}
    total::T
end

"""Fit `KernelCenter` object"""
function fit(::Type{KernelCenter}, K::AbstractMatrix{<:Real})
    n = size(K, 1)
    means = vec(mean(K, dims=2))
    KernelCenter(means, sum(means) / n)
end

"""Center kernel matrix."""
function transform!(C::KernelCenter, K::AbstractMatrix{<:Real})
    r, c = size(K)
    tot = C.total
    means = mean(K, dims=1)
    @simd for i in 1:r
        for j in 1:c
            @inbounds K[i, j] -= C.means[i] + means[j] - tot
        end
    end
    return K
end

"""
This type contains kernel PCA model parameters.
"""
struct KernelPCA{T<:Real} <: NonlinearDimensionalityReduction
    X::AbstractMatrix{T}           # fitted data or precomputed kernel
    ker::Union{Nothing, Function}  # kernel function
    center::KernelCenter           # kernel center
    λ::AbstractVector{T}           # eigenvalues  in feature space
    α::AbstractMatrix{T}           # eigenvectors in feature space
    inv::AbstractMatrix{T}         # inverse transform coefficients
end

## properties
"""
    size(M::KernelPCA)

Returns a tuple with the input dimension ``d``, *i.e* the dimension of the observation space, and the output dimension ``p``, *i.e* the dimension of the principal subspace.
"""
size(M::KernelPCA) = (size(M.X, 1), length(M.λ))

"""
    eigvals(M::KernelPCA)

Return eigenvalues of the kernel matrix of the model `M`.
"""
eigvals(M::KernelPCA) = M.λ

"""
    eigvecs(M::KernelPCA)

Return eigenvectors of the kernel matrix of the model `M`.
"""
eigvecs(M::KernelPCA) = M.α

"""
    projection(M::KernelPCA)

Return the projection matrix (of size ``n \\times p``).
Each column of the projection matrix corresponds to an eigenvector, and ``n`` is a number of observations.

The principal components are arranged in descending order of the corresponding eigenvalues.
"""
projection(M::KernelPCA) = eigvecs(M) ./ sqrt.(eigvals(M)')

## use

"""
    predict(M::KernelPCA, x)

Transform out-of-sample transformation of `x` into a kernel space of the model `M`.

Here, `x` can be either a vector of length `d` or a matrix where each column is an observation.
"""
function predict(M::KernelPCA, x::AbstractVecOrMat{T}) where {T<:Real}
    k = pairwise(M.ker, eachcol(M.X), eachcol(x))
    transform!(M.center, k)
    return projection(M)'*k
end


"""
    predict(M::KernelPCA)

Transform the data fitted to the model `M` to a kernel space of the model.
"""
predict(M::KernelPCA) = sqrt.(eigvals(M)) .* eigvecs(M)'


"""
    reconstruct(M::KernelPCA, y)

Approximately reconstruct observations, given in `y`, to the original space using the kernel PCA model `M`.

Here, `y` can be either a vector of length `p` or a matrix where each column gives the principal components for an observation.
"""
function reconstruct(M::KernelPCA, y::AbstractVecOrMat{T}) where {T<:Real}
    if size(M.inv, 1) == 0
        throw(ArgumentError("Inverse transformation coefficients are not available, set `inverse` parameter when fitting data"))
    end
    Pᵗ = predict(M)
    k = pairwise(M.ker, eachcol(Pᵗ), eachcol(y))
    return M.inv*k
end

## show

function show(io::IO, M::KernelPCA)
    indim, outdim = size(M)
    print(io, "Kernel PCA(indim = $indim, outdim = $outdim)")
end

## interface functions
"""
    fit(KernelPCA, X; ...)

Perform kernel PCA over the data given in a matrix `X`. Each column of `X` is an observation.

This method returns an instance of [`KernelPCA`](@ref).

**Keyword arguments:**

Let `(d, n) = size(X)` be respectively the input dimension and the number of observations:

- `kernel`: The kernel function. This functions accepts two vector arguments `x` and `y`,
and returns a scalar value (*default:* `(x,y)->x'y`)
- `solver`: The choice of solver:
    - `:eig`: uses `LinearAlgebra.eigen` (*default*)
    - `:eigs`: uses `Arpack.eigs` (always used for sparse data)
- `maxoutdim`:  Maximum output dimension (*default* `min(d, n)`)
- `inverse`: Whether to perform calculation for inverse transform for non-precomputed kernels (*default* `false`)
- `β`: Hyperparameter of the ridge regression that learns the inverse transform (*default* `1` when `inverse` is `true`).
- `tol`: Convergence tolerance for `eigs` solver (*default* `0.0`)
- `maxiter`: Maximum number of iterations for `eigs` solver (*default* `300`)
"""
function fit(::Type{KernelPCA}, X::AbstractMatrix{T};
             kernel::Union{Nothing, Function} = (x,y)->x'y,
             maxoutdim::Int = min(size(X)...),
             remove_zero_eig::Bool = false, atol::Real = 1e-10,
             solver::Symbol = :eig,
             inverse::Bool = false,  β::Real = convert(T, 1.0),
             tol::Real = 0.0, maxiter::Real = 300) where {T<:Real}

    d, n = size(X)
    maxoutdim = min(min(d, n), maxoutdim)

    # set kernel function if available
    K = if isa(kernel, Function)
        pairwise(kernel, eachcol(X), symmetric=true)
    elseif kernel === nothing
        @assert issymmetric(X) "Precomputed kernel matrix must be symmetric."
        inverse = false
        copy(X)
    else
        throw(ArgumentError("Incorrect kernel type. Use a function or a precomputed kernel."))
    end

    # center kernel
    center = fit(KernelCenter, K)
    transform!(center, K)

    # perform eigenvalue decomposition
    evl, evc = if solver == :eigs || SparseArrays.issparse(K)
        evl, evc = eigs(K, nev=maxoutdim, which=:LR, v0=2.0*rand(n) .- 1.0, tol=tol, maxiter=maxiter)
        real.(evl), real.(evc)
    else
        Eg = eigen(Hermitian(K))
        Eg.values, Eg.vectors
    end

    # sort eigenvalues in descending order
    ord = sortperm(evl; rev=true)
    ord = ord[1:min(length(ord), maxoutdim)]

    # remove zero eigenvalues
    λ, α = if remove_zero_eig
        ez = map(!, isapprox.(evl[ord], zero(T), atol=atol))
        evl[ord[ez]], evc[:, ord[ez]]
    else
        evl[ord], evc[:, ord]
    end

    # calculate inverse transform coefficients
    Q = zeros(T, 0, 0)
    if inverse
        Pᵗ = α' .* sqrt.(λ)
        KT = pairwise(kernel, eachcol(Pᵗ), symmetric=true)
        Q = (KT + diagm(0 => fill(β, size(KT,1)))) \ X'
    end

    KernelPCA(X, kernel, center, λ, α, Q')
end
