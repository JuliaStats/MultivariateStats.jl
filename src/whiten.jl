# Whitening

## Solve whitening based on covariance
#
# finds W, such that W'CW = I
#
function cov_whitening{T<:AbstractFloat}(C::Cholesky{T})
    cf = C[:UL]
    if VERSION < v"0.4.0-"
        full(istriu(cf) ? inv(Triangular(C.UL, :U)) : inv(Triangular(C.UL, :L)'))::Matrix{T}
    else
        full(inv(istriu(cf) ? cf : cf'))::Matrix{T}
    end
end

cov_whitening!{T<:AbstractFloat}(C::DenseMatrix{T}) = cov_whitening(cholfact!(C, :U))
cov_whitening{T<:AbstractFloat}(C::DenseMatrix{T}) = cov_whitening!(copy(C))

cov_whitening!{T<:AbstractFloat}(C::DenseMatrix{T}, regcoef::Real) =
    cov_whitening!(regularize_symmat!(C, regcoef))

cov_whitening{T<:AbstractFloat}(C::DenseMatrix{T}, regcoef::Real) =
    cov_whitening!(copy(C), regcoef)


## Whitening type

immutable Whitening{T<:AbstractFloat}
    mean::Vector{T}
    W::Matrix{T}
end

function Whitening{T<:AbstractFloat}(mean::Vector{T}, W::Matrix{T})
    d, d2 = size(W)
    d == d2 || error("W must be a square matrix.")
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Sizes of mean and W are inconsistent."))
    return Whitening{T}(mean, W)
end

indim(f::Whitening) = size(f.W, 1)
outdim(f::Whitening) = size(f.W, 2)
Base.mean(f::Whitening) = fullmean(indim(f), f.mean)

transform(f::Whitening, x::AbstractVecOrMat) = At_mul_B(f.W, centralize(x, f.mean))

## Fit whitening to data

function fit{T<:AbstractFloat}(::Type{Whitening}, X::DenseMatrix{T};
                               mean=nothing, regcoef::Real=zero(T))
    n = size(X, 2)
    n > 1 || error("X must contain more than one sample.")
    mv = preprocess_mean(X, mean)
    Z = centralize(X, mv)
    C = scale!(A_mul_Bt(Z, Z), one(T) / (n - 1))
    return Whitening(mv, cov_whitening!(C, regcoef))
end

# invsqrtm

function _invsqrtm!{T<:AbstractFloat}(C::Matrix{T})
    n = size(C, 1)
    size(C, 2) == n || error("C must be a square matrix.")
    E = eigfact!(Symmetric(C))
    U = E.vectors
    evs = E.values
    for i = 1:n
        @inbounds evs[i] = 1.0 / sqrt(sqrt(evs[i]))
    end
    scale!(U, evs)
    return A_mul_Bt(U, U)
end

invsqrtm{T<:AbstractFloat}(C::DenseMatrix{T}) = _invsqrtm!(copy(C))

