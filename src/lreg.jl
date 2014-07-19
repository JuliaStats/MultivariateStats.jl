# Ridge Regression (Tikhonov regularization)

## auxiliary

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
    _llsq(X, Y, trans, bias)
end

function _llsq{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, trans::Bool, bias::Bool)
    if bias
        if trans
            X_ = _vaug(X)
            A = cholfact!(A_mul_Bt(X_, X_)) \ (X_ * Y)
        else
            X_ = _haug(X)
            A = cholfact!(X_'X_) \ (X_'Y)
        end
    else
        if trans
            A = cholfact!(A_mul_Bt(X, X)) \ (X * Y)
        else
            A = cholfact!(X'X) \ (X'Y)
        end
    end
    return A::lrsoltype(Y)
end


## ridge regression

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::Real; 
                                trans::Bool=false, bias::Bool=true)
    lreg_chkdims(X, Y, trans)
    _ridge(X, Y, convert(T, r), trans, bias)
end

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::DenseVector{T}; 
                                trans::Bool=false, bias::Bool=true)
    d = lreg_chkdims(X, Y, trans)
    length(r) == d || throw("Incorrect length of r.")
    _ridge(X, Y, r, trans, bias)
end

function ridge{T<:FloatingPoint}(X::DenseMatrix{T}, Y::DenseVecOrMat{T}, r::DenseMatrix{T}; 
                                trans::Bool=false, bias::Bool=true)
    d = lreg_chkdims(X, Y, trans)
    size(r) == (d, d) || throw("Incorrect size of r.")
    _ridge(X, Y, r, trans, bias)
end

function _ridge(X::DenseMatrix, Y::DenseVecOrMat, r::Real, trans::Bool, bias::Bool)

end

