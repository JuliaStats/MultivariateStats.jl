# Ridge Regression (Tikhonov regularization)

#### auxiliary

function lreg_chkdims(X::AbstractMatrix, Y::AbstractVecOrMat, trans::Bool)
    mX, nX = size(X)
    dX = ifelse(trans, mX, nX)
    dY = ifelse(trans, nX, mX)
    size(Y, 1) == dY || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
    return dX
end

lrsoltype{T}(::DenseVector{T}) = Vector{T}
lrsoltype{T}(::DenseMatrix{T}) = Matrix{T}

_vaug{T}(X::DenseMatrix{T}) = vcat(X, ones(T, 1, size(X,2)))::Matrix{T}
_haug{T}(X::DenseMatrix{T}) = hcat(X, ones(T, size(X,1), 1))::Matrix{T}


## linear least square

function llsq{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}; 
                                trans::Bool=false, bias::Bool=true)
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

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::Real; 
                                trans::Bool=false, bias::Bool=true)
    lreg_chkdims(X, Y, trans)
    r >= zero(r) || error("r must be non-negative.")
    _ridge(X, Y, convert(T, r), trans, bias)
end

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::DenseVector{T}; 
                                trans::Bool=false, bias::Bool=true)
    d = lreg_chkdims(X, Y, trans)
    length(r) == d || throw(DimensionMismatch("Incorrect length of r."))
    _ridge(X, Y, r, trans, bias)
end

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::DenseMatrix{T}; 
                                trans::Bool=false, bias::Bool=true)
    d = lreg_chkdims(X, Y, trans)
    size(r) == (d, d) || throw(DimensionMismatch("Incorrect size of r."))
    _ridge(X, Y, r, trans, bias)
end

## implementation

function _ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, 
                                  r::Union(Real, DenseVecOrMat), trans::Bool, bias::Bool)
    if bias
        if trans
            X_ = _vaug(X)
            A = cholfact!(_ridge_reg!(A_mul_Bt(X_, X_), r, bias)) \ (X_ * Y)
        else
            X_ = _haug(X)
            A = cholfact!(_ridge_reg!(X_'X_, r, bias)) \ (X_'Y)
        end
    else
        if trans
            A = cholfact!(_ridge_reg!(A_mul_Bt(X, X), r, bias)) \ (X * Y)
        else
            A = cholfact!(_ridge_reg!(X'X, r, bias)) \ (X'Y)
        end
    end
    return A::lrsoltype(Y)
end

function _ridge_reg!(Q::Matrix, r::Real, bias::Bool)
    if r > zero(r)
        n = size(Q, 1) - int(bias)
        for i = 1:n
            @inbounds Q[i,i] += r
        end
    end
    return Q
end

function _ridge_reg!(Q::Matrix, r::DenseVector, bias::Bool)
    n = size(Q, 1) - int(bias)
    @assert length(r) == n
    for i = 1:n
        @inbounds Q[i,i] += r[i]
    end
    return Q
end

function _ridge_reg!(Q::Matrix, r::Matrix, bias::Bool)
    n = size(Q, 1) - int(bias)
    @assert size(r) == (n, n)
    for j = 1:n, i = 1:n
        @inbounds Q[i,j] += r[i,j]
    end
    return Q
end

