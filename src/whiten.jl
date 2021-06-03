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

"""
    length(f)

Get the dimension of the  whitening transform `f`.
"""
length(f::Whitening) = size(f.W, 1)

"""
    size(f)

Dimensions of the coefficient matrix of the whitening transform `f`.
"""
size(f::Whitening) = size(f.W)

"""
    mean(f)

Get the mean vector of the whitening transformation `f`.

**Note:** if mean is empty, this function returns a zero vector of `length(f)`.
"""
mean(f::Whitening) = fullmean(length(f), f.mean)


"""
    transform(f, x)

Apply the whitening transform `f` to a vector or a matrix `x` with samples in columns, as ``\\mathbf{W}^T (\\mathbf{x} - \\boldsymbol{\\mu})``.
"""
function transform(f::Whitening, x::AbstractVecOrMat{<:Real})
    s = size(x)
    Z, dims = if length(s) == 1
        length(f.mean) == s[1] || throw(DimensionMismatch("Inconsistent dimensions."))
        x - f.mean, 2
    else
        dims = (s[1] == length(f.mean)) + 1
        length(f.mean) == s[3-dims] || throw(DimensionMismatch("Inconsistent dimensions."))
        x .- (dims == 2 ? f.mean : transpose(f.mean)), dims
    end
    if dims == 2
        transpose(f.W) * Z
    else
        Z * f.W
    end
end

"""
    fit(Whitening, X::AbstractMatrix{T}; kwargs...)

Estimate a whitening transform from the data given in `X`.

This function returns an instance of [`Whitening`](@ref)

**Keyword Arguments:**
- `regcoef`: The regularization coefficient. The covariance will be regularized as follows when `regcoef` is positive `C + (eigmax(C) * regcoef) * eye(d)`. Default values is `zero(T)`.

- `dims`: if `1` the transformation calculated from the row samples. fit standardization parameters in column-wise fashion;
  if `2` the transformation calculated from the column samples. The default is `nothing`, which is equivalent to `dims=2` with a deprecation warning.

- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (**default**)
    - a pre-computed mean vector

**Note:** This function internally relies on [`cov_whitening`](@ref) to derive the transformation `W`.
"""
function fit(::Type{Whitening}, X::AbstractMatrix{T};
             dims::Union{Integer,Nothing}=nothing,
             mean=nothing, regcoef::Real=zero(T)) where {T<:Real}
    if dims === nothing
        Base.depwarn("fit(Whitening, x) is deprecated: use fit(Whitening, x, dims=2) instead", :fit)
        dims = 2
    end
    if dims == 1
        n = size(X,1)
        n >= 2 || error("X must contain at least two rows.")
    elseif dims == 2
        n = size(X, 2)
        n >= 2 || error("X must contain at least two columns.")
    else
        throw(DomainError(dims, "fit only accept dims to be 1 or 2."))
    end
    mv = preprocess_mean(X, mean; dims=dims)
    Z = centralize((dims==1 ? transpose(X) : X), mv)
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
