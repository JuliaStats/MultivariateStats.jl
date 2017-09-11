# Kernel Principal Component Analysis

"""Center a kernel matrix"""
immutable KernelCenter{T<:AbstractFloat}
    means::AbstractVector{T}
    total::T
end

"""Fit `KernelCenter` object"""
function fit{T<:AbstractFloat}(::Type{KernelCenter}, K::AbstractMatrix{T})
    n = size(K,1)
    means = vec(mean(K,2))
    KernelCenter(means, sum(means)/n)
end

"""Center kernel matrix."""
function transform!{T<:AbstractFloat}(C::KernelCenter{T}, K::AbstractMatrix{T})
    n, m = size(K)
    @simd for i in 1:n
        for j in 1:m
            @inbounds K[i,j] -= C.means[i] + C.means[j] - C.total
        end
    end
    return K
end

"""Kernel PCA type"""
immutable KernelPCA{T<:AbstractFloat}
    X::AbstractMatrix{T}  # fitted data
    ker::Function         # kernel function
    center::KernelCenter  # kernel center
    λ::DenseVector{T}     # eigenvalues  in feature space
    α::DenseMatrix{T}     # eigenvectors in feature space
end

## properties

indim(M::KernelPCA) = size(M.X, 1)
outdim(M::KernelPCA) = length(M.λ)

projection(M::KernelPCA) = M.α
principalvars(M::KernelPCA) = M.λ

## use

function transform{T<:AbstractFloat}(M::KernelPCA{T}, x::AbstractVecOrMat{T})
    k = MultivariateStats.pairwise(M.ker, M.X, x)
    transform!(M.center, k)
    return (M.α')./sqrt.(M.λ)*k
end

## show

function Base.show(io::IO, M::KernelPCA)
    print(io, "Kernel PCA(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## core algorithms

function pairwise!{T<:AbstractFloat}(K::AbstractVecOrMat{T}, kernel::Function,
    X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    n = size(X, 2)
    m = size(Y, 2)
    for j = 1:m
        aj = view(Y, :, j)
        for i = j:n
            @inbounds K[i, j] = kernel(view(X, :, i), aj)
        end
        j <= n && for i = 1:(j - 1)
            @inbounds K[i, j] = K[j, i]   # leveraging the symmetry
        end
    end
    K
end

pairwise!{T<:AbstractFloat}(K::AbstractVecOrMat{T}, kernel::Function, X::AbstractVecOrMat{T}) =
    pairwise!(K, kernel, X, X)

function pairwise{T<:AbstractFloat}(kernel::Function, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    n = size(X, 2)
    m = size(Y, 2)
    K = similar(X, n, m)
    pairwise!(K, kernel, X, Y)
end

pairwise{T<:AbstractFloat}(kernel::Function, X::AbstractVecOrMat{T}) =
    pairwise(kernel, X, X)

## interface functions

function fit{T<:AbstractFloat}(::Type{KernelPCA}, X::AbstractMatrix{T};
                               kernel::Function=(x,y)->x'*y,
                               maxoutdim::Int=min(size(X)...),
                               remove_zero_eig::Bool=false,
                               solver::Symbol = :eig,
                               tol::Real = 1e-12, tot::Real = 300)
    d, n = size(X)
    maxoutdim = min(min(d, n), maxoutdim)

    K = similar(X, n, n)
    pairwise!(K, kernel, X)
    center = fit(KernelCenter, K)
    transform!(center, K)

    evl, evc = if solver == :eigs || issparse(K)
        evl, evc = eigs(K, nev=maxoutdim, which=:LR, v0=2.0*rand(n)-1.0, tol=tol, maxiter=tot)
        real.(evl), real.(evc)
    else
        Eg = eigfact(Hermitian(K))
        Eg[:values], Eg[:vectors]
    end

    # sort eigenvalues in descending order
    ord = sortperm(evl; rev=true)[1:maxoutdim]

    # remove zero eigenvalues
    λ, α = if remove_zero_eig
        ez = .!isapprox.(evl[ord], zero(T), atol=tol)
        evl[ord[ez]], evc[:, ord[ez]]
    else
        evl[ord], evc[:, ord]
    end

    KernelPCA(X, kernel, center, λ, α)
end
