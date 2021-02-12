# Kernel Principal Component Analysis

import Arpack

"""Center a kernel matrix"""
struct KernelCenter{T<:Real}
    means::AbstractVector{T}
    total::T
end

"""Fit `KernelCenter` object"""
function fit(::Type{KernelCenter}, K::AbstractMatrix{T}) where {T<:Real}
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

"""Kernel PCA type"""
struct KernelPCA{T<:Real}
    X::AbstractMatrix{T}           # fitted data or precomputed kernel
    ker::Union{Nothing, Function}  # kernel function
    center::KernelCenter           # kernel center
    λ::AbstractVector{T}           # eigenvalues  in feature space
    α::AbstractMatrix{T}           # eigenvectors in feature space
    inv::AbstractMatrix{T}         # inverse transform coefficients
end

## properties

indim(M::KernelPCA) = size(M.X, 1)
outdim(M::KernelPCA) = length(M.λ)

projection(M::KernelPCA) = M.α ./ sqrt.(M.λ')
principalvars(M::KernelPCA) = M.λ

## use

"""Calculate transformation to kernel space"""
function transform(M::KernelPCA, x::AbstractVecOrMat{<:Real})
    k = pairwise(M.ker, M.X, x)
    transform!(M.center, k)
    return projection(M)'*k
end

transform(M::KernelPCA) = sqrt.(M.λ) .* M.α'

"""Calculate inverse transformation to original space"""
function reconstruct(M::KernelPCA, y::AbstractVecOrMat{<:Real})
    if size(M.inv, 1) == 0
        throw(ArgumentError("Inverse transformation coefficients are not available, set `inverse` parameter when fitting data"))
    end
    Pᵗ = M.α' .* sqrt.(M.λ)
    k = pairwise(M.ker, Pᵗ, y)
    return M.inv*k
end

## show

function Base.show(io::IO, M::KernelPCA)
    print(io, "Kernel PCA(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## interface functions

function fit(::Type{KernelPCA}, X::AbstractMatrix{T};
             kernel::Union{Nothing, Function} = (x,y)->x'y,
             maxoutdim::Int = min(size(X)...),
             remove_zero_eig::Bool = false, atol::Real = 1e-10,
             solver::Symbol = :eig,
             inverse::Bool = false,  β::Real = 1.0,
             tol::Real = 0.0, maxiter::Real = 300) where {T<:Real}

    d, n = size(X)
    maxoutdim = min(min(d, n), maxoutdim)

    # set kernel function if available
    K = if isa(kernel, Function)
        pairwise(kernel, X)
    elseif kernel === nothing
        @assert issymmetric(X) "Precomputed kernel matrix must be symmetric."
        inverse = false
        X
    else
        throw(ArgumentError("Incorrect kernel type. Use a function or a precomputed kernel."))
    end

    # center kernel
    center = fit(KernelCenter, K)
    transform!(center, K)

    # perform eigenvalue decomposition
    evl, evc = if solver == :eigs || SparseArrays.issparse(K)
        evl, evc = Arpack.eigs(K, nev=maxoutdim, which=:LR, v0=2.0*rand(n) .- 1.0, tol=tol, maxiter=maxiter)
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
        KT = pairwise(kernel, Pᵗ)
        Q = (KT + diagm(0 => fill(β, size(KT,1)))) \ X'
    end

    KernelPCA(X, kernel, center, λ, α, Q')
end
