# Whitening

## Solve whitening based on covariance
#
# finds W, such that W'CW = I
#
function cov_whitening(C::Cholesky{T}) where T<:AbstractFloat
    cf = C.UL
    Matrix{T}(inv(istriu(cf) ? cf : cf'))
end

cov_whitening!(C::DenseMatrix{T}) where T<:AbstractFloat = cov_whitening(cholesky!(Hermitian(C, :U)))
cov_whitening(C::DenseMatrix{T}) where T<:AbstractFloat = cov_whitening!(copy(C))

cov_whitening!(C::DenseMatrix{T}, regcoef::Real) where T<:AbstractFloat =
    cov_whitening!(regularize_symmat!(C, regcoef))

cov_whitening(C::DenseMatrix{T}, regcoef::Real) where T<:AbstractFloat =
    cov_whitening!(copy(C), regcoef)


## Whitening type

struct Whitening{T<:AbstractFloat}
    mean::Vector{T}
    W::Matrix{T}

    function Whitening{T}(mean::Vector{T}, W::Matrix{T}) where T<:AbstractFloat
        d, d2 = size(W)
        d == d2 || error("W must be a square matrix.")
        isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Sizes of mean and W are inconsistent."))
        return new(mean, W)
    end
end
Whitening(mean::Vector{T}, W::Matrix{T}) where T<:AbstractFloat = Whitening{T}(mean, W)

indim(f::Whitening) = size(f.W, 1)
outdim(f::Whitening) = size(f.W, 2)
mean(f::Whitening) = fullmean(indim(f), f.mean)

transform(f::Whitening, x::AbstractVecOrMat) = transpose(f.W) * centralize(x, f.mean)

## Fit whitening to data

function fit(::Type{Whitening}, X::DenseMatrix{T};
             mean=nothing, regcoef::Real=zero(T)) where T<:AbstractFloat
    n = size(X, 2)
    n > 1 || error("X must contain more than one sample.")
    mv = preprocess_mean(X, mean)
    Z = centralize(X, mv)
    C = rmul!(Z * transpose(Z), one(T) / (n - 1))
    return Whitening(mv, cov_whitening!(C, regcoef))
end

# invsqrtm

function _invsqrtm!(C::Matrix{T}) where T<:AbstractFloat
    n = size(C, 1)
    size(C, 2) == n || error("C must be a square matrix.")
    E = eigen!(Symmetric(C))
    U = E.vectors
    evs = E.values
    for i = 1:n
        @inbounds evs[i] = 1.0 / sqrt(sqrt(evs[i]))
    end
    rmul!(U, Diagonal(evs))
    return U * transpose(U)
end

invsqrtm(C::DenseMatrix{T}) where T<:AbstractFloat = _invsqrtm!(copy(C))
