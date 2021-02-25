# Whitening

"""
    cov_whitening(C)

Derive the whitening transform coefficient matrix `W` given the covariance matrix `C`. Here, `C` can be either a square matrix, or an instance of `Cholesky`.

Internally, this function solves the whitening transform using Cholesky factorization. The rationale is as follows: let ``\\mathbf{C} = \\mathbf{U}^T \\mathbf{U}`` and ``\\mathbf{W} = \\mathbf{U}^{-1}``, then ``\\mathbf{W}^T \\mathbf{C} \\mathbf{W} = \\mathbf{I}``.

**Note:** The return matrix `W` is an upper triangular matrix.
"""
function cov_whitening(C::Cholesky{T}) where {T<:Real}
    cf = C.UL
    Matrix{T}(inv(istriu(cf) ? cf : cf'))
end

"""
    cov_whitening!(C)

In-place version of `cov_whitening(C)`, in which the input matrix `C` will be overwritten during computation. This can be more efficient when `C` is no longer used.
"""
cov_whitening!(C::AbstractMatrix{<:Real}) = cov_whitening(cholesky!(Hermitian(C, :U)))
cov_whitening(C::AbstractMatrix{<:Real}) = cov_whitening!(copy(C))

"""
    cov_whitening!(C, regcoef)

In-place version of `cov_whitening(C, regcoef)`, in which the input matrix `C` will be overwritten during computation. This can be more efficient when `C` is no longer used.
"""
cov_whitening!(C::AbstractMatrix{<:Real}, regcoef::Real) = cov_whitening!(regularize_symmat!(C, regcoef))

"""
    cov_whitening(C, regcoef)

Derive a whitening transform based on a regularized covariance, as `C + (eigmax(C) * regcoef) * eye(d)`.
"""
cov_whitening(C::AbstractMatrix{<:Real}, regcoef::Real) = cov_whitening!(copy(C), regcoef)

## Whitening type

"""
A whitening transform representation.
"""
struct Whitening{T<:Real} <: AbstractDataTransform
    mean::AbstractVector{T}
    W::AbstractMatrix{T}

    function Whitening{T}(mean::AbstractVector{T}, W::AbstractMatrix{T}) where {T<:Real}
        d, d2 = size(W)
        d == d2 || error("W must be a square matrix.")
        isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Sizes of mean and W are inconsistent."))
        return new(mean, W)
    end
end
Whitening(mean::AbstractVector{T}, W::AbstractMatrix{T}) where {T<:Real} = Whitening{T}(mean, W)

indim(f::Whitening) = size(f.W, 1)
outdim(f::Whitening) = size(f.W, 2)

"""
    size(f)

Dimensions of the coefficient matrix of the whitening transform `f`.
"""
size(f::Whitening) = size(f.W)

"""
    mean(f)

Get the mean vector of the whitening transformation `f`.

**Note:** if mean is empty, this function returns a zero vector of length [`outdim`](@ref) .
"""
mean(f::Whitening) = fullmean(indim(f), f.mean)


"""
    transform(f, x)

Apply the whitening transform `f` to a vector or a matrix `x` with samples in columns, as ``\\mathbf{W}^T (\\mathbf{x} - \\boldsymbol{\\mu})``.
"""
transform(f::Whitening, x::AbstractVecOrMat{<:Real}) = transpose(f.W) * centralize(x, f.mean)

"""
    fit(::Type{Whitening},  X::AbstractMatrix{T}; kwargs...)

Estimate a whitening transform from the data given in `X`. Here, `X` should be a matrix, whose columns give the samples.

This function returns an instance of [`Whitening`](@ref)

**Keyword Arguments:**
- `regcoef`: The regularization coefficient. The covariance will be regularized as follows when `regcoef` is positive `C + (eigmax(C) * regcoef) * eye(d)`. Default values is `zero(T)`.

- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (**default**)
    - a pre-computed mean vector

**Note:** This function internally relies on [`cov_whitening`](@ref) to derive the transformation `W`.
"""
function fit(::Type{Whitening}, X::AbstractMatrix{T};
             mean=nothing, regcoef::Real=zero(T)) where {T<:Real}
    n = size(X, 2)
    n > 1 || error("X must contain more than one sample.")
    mv = preprocess_mean(X, mean)
    Z = centralize(X, mv)
    C = rmul!(Z * transpose(Z), one(T) / (n - 1))
    return Whitening(mv, cov_whitening!(C, regcoef))
end

# invsqrtm

function _invsqrtm!(C::AbstractMatrix{<:Real})
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

"""
    invsqrtm(C)

Compute `inv(sqrtm(C))` through symmetric eigenvalue decomposition.
"""
invsqrtm(C::AbstractMatrix{<:Real}) = _invsqrtm!(copy(C))
