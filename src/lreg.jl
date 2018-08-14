# Ridge Regression (Tikhonov regularization)

#### auxiliary

function lreg_chkdims(X::AbstractMatrix, Y::AbstractVecOrMat, trans::Bool)
    mX, nX = size(X)
    dX = ifelse(trans, mX, nX)
    dY = ifelse(trans, nX, mX)
    size(Y, 1) == dY || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
    return dX
end

lrsoltype(::AbstractVector{T}) where T = Vector{T}
lrsoltype(::AbstractMatrix{T}) where T = Matrix{T}

_vaug(X::AbstractMatrix{T}) where T = vcat(X, ones(T, 1, size(X,2)))::Matrix{T}
_haug(X::AbstractMatrix{T}) where T = hcat(X, ones(T, size(X,1), 1))::Matrix{T}


## linear least square

function llsq(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T};
              trans::Bool=false, bias::Bool=true) where T<:AbstractFloat
    if trans
        mX, nX = size(X)
        size(Y, 1) == nX || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
        mX <= nX || error("mX <= nX is required when trans is false.")
    else
        mX, nX = size(X)
        size(Y, 1) == mX || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
        mX >= nX || error("mX >= nX is required when trans is false.")
    end
    _ridge(X, Y, zero(T), trans, bias)
end

## ridge regression

function ridge(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T}, r::Real;
               trans::Bool=false, bias::Bool=true) where T<:AbstractFloat
    lreg_chkdims(X, Y, trans)
    r >= zero(r) || error("r must be non-negative.")
    _ridge(X, Y, convert(T, r), trans, bias)
end

function ridge(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T}, r::AbstractVector{T};
               trans::Bool=false, bias::Bool=true) where T<:AbstractFloat
    d = lreg_chkdims(X, Y, trans)
    length(r) == d || throw(DimensionMismatch("Incorrect length of r."))
    _ridge(X, Y, r, trans, bias)
end

function ridge(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T}, r::AbstractMatrix{T};
               trans::Bool=false, bias::Bool=true) where T<:AbstractFloat
    d = lreg_chkdims(X, Y, trans)
    size(r) == (d, d) || throw(DimensionMismatch("Incorrect size of r."))
    _ridge(X, Y, r, trans, bias)
end

## implementation

function _ridge(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T},
                r::Union{Real, AbstractVecOrMat}, trans::Bool, bias::Bool) where T<:AbstractFloat
    if bias
        if trans
            X_ = _vaug(X)
            A = cholesky!(Hermitian(_ridge_reg!(X_ * transpose(X_), r, bias))) \ (X_ * Y)
        else
            X_ = _haug(X)
            A = cholesky!(Hermitian(_ridge_reg!(X_'X_, r, bias))) \ (X_'Y)
        end
    else
        if trans
            A = cholesky!(Hermitian(_ridge_reg!(X * X', r, bias))) \ (X * Y)
        else
            A = cholesky!(Hermitian(_ridge_reg!(X'X, r, bias))) \ (X'Y)
        end
    end
    return A::lrsoltype(Y)
end

function _ridge_reg!(Q::Matrix, r::Real, bias::Bool)
    if r > zero(r)
        n = size(Q, 1) - Int(bias)
        for i = 1:n
            @inbounds Q[i,i] += r
        end
    end
    return Q
end

function _ridge_reg!(Q::AbstractMatrix, r::AbstractVector, bias::Bool)
    n = size(Q, 1) - Int(bias)
    @assert length(r) == n
    for i = 1:n
        @inbounds Q[i,i] += r[i]
    end
    return Q
end

function _ridge_reg!(Q::AbstractMatrix, r::AbstractMatrix, bias::Bool)
    n = size(Q, 1) - Int(bias)
    @assert size(r) == (n, n)
    for j = 1:n, i = 1:n
        @inbounds Q[i,j] += r[i,j]
    end
    return Q
end
