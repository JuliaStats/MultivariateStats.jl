# Regression

## Auxiliary

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


## Linear Least Square Regression


"""
    llsq(X, y; ...)

Solve the linear least square problem.

Here, `y` can be either a vector, or a matrix where each column is a response vector.

This function accepts two keyword arguments:

- `dims`: whether input observations are stored as rows (`1`) or columns (`2`). (default is `1`)
- `bias`: whether to include the bias term `b`. (default is `true`)

The function results the solution `a`. In particular, when `y` is a vector (matrix), `a` is also a vector (matrix). If `bias` is true, then the returned array is augmented as `[a; b]`.
"""
function llsq(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T};
              trans::Bool=false, bias::Bool=true,
              dims::Union{Integer,Nothing}=nothing) where {T<:Real}
    if dims === nothing && trans
        Base.depwarn("`trans` argument is deprecated, use llsq(X, Y, dims=d) instead.", :trans)
        dims = 1
    end
    if dims == 2
        mX, nX = size(X)
        size(Y, 1) == nX || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
        mX <= nX || error("mX <= nX is required when trans is false.")
    else
        mX, nX = size(X)
        size(Y, 1) == mX || throw(DimensionMismatch("Dimensions of X and Y mismatch."))
        mX >= nX || error("mX >= nX is required when trans is false.")
    end
    _ridge(X, Y, zero(T), dims == 2, bias)
end
llsq(x::AbstractVector{T}, y::AbstractVector{T}; kwargs...) where {T<:Real} =
    llsq(x[:,:], y; dims=1, kwargs...)

## Ridge Regression (Tikhonov regularization)

"""

    ridge(X, y, r; ...)

Solve the ridge regression problem.

Here, ``y`` can be either a vector, or a matrix where each column is a response vector.

The argument `r` gives the quadratic regularization matrix ``Q``, which can be in either of the following forms:

- `r` is a real scalar, then ``Q`` is considered to be `r * eye(n)`, where `n` is the dimension of `a`.
- `r` is a real vector, then ``Q`` is considered to be `diagm(r)`.
- `r` is a real symmetric matrix, then ``Q`` is simply considered to be `r`.

This function accepts two keyword arguments:

- `dims`: whether input observations are stored as rows (`1`) or columns (`2`). (default is `1`)
- `bias`: whether to include the bias term `b`. (default is `true`)

The function results the solution `a`. In particular, when `y` is a vector (matrix), `a` is also a vector (matrix). If `bias` is true, then the returned array is augmented as `[a; b]`.
"""
function ridge(X::AbstractMatrix{T}, Y::AbstractVecOrMat{T}, r::Union{Real, AbstractVecOrMat};
               trans::Bool=false, bias::Bool=true,
               dims::Union{Integer,Nothing}=nothing) where {T<:Real}
    if dims === nothing && trans
        Base.depwarn("`trans` argument is deprecated, use ridge(X, Y, r, dims=d) instead.", :trans)
        dims = 1
    end
    d = lreg_chkdims(X, Y, dims == 2)
    if isa(r, Real)
        r >= zero(r) || error("r must be non-negative.")
        r = convert(T <: AbstractFloat ? T : Float64, r)
    elseif isa(r, AbstractVector)
        length(r) == d || throw(DimensionMismatch("Incorrect length of r."))
    elseif isa(r, AbstractMatrix)
        size(r) == (d, d) || throw(DimensionMismatch("Incorrect size of r."))
    end
    _ridge(X, Y, r, dims == 2, bias)
end
ridge(x::AbstractVector{T}, y::AbstractVector{T}, r::Union{Real, AbstractVecOrMat};
      kwargs...) where {T<:Real} = ridge(x[:,:], y, r; dims=1, kwargs...)

### implementation

function _ridge(_X::AbstractMatrix{T}, _Y::AbstractVecOrMat{T},
                r::Union{Real, AbstractVecOrMat}, trans::Bool, bias::Bool) where {T<:Real}
    # convert integer data to Float64
    X = T <: AbstractFloat ? _X : convert(Array{Float64}, _X)
    Y = T <: AbstractFloat ? _Y : convert(Array{Float64}, _Y)
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

## Isotonic Regression

"""
    isotonic(x, y[, w])

Solve the isotonic regression problem using the pool adjacent violators algorithm[^1].

Here `x` is the regressor vector, `y` is response vector, and `w` is an optional
weights vector.

The function returns a prediction vector of the same size as the regressor vector `x`.
"""
function isotonic(x::AbstractVector{T}, y::AbstractVector{T},
                  w::AbstractVector{T} = ones(T, length(y))) where {T<:Real}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("Dimensions of x and y mismatch."))

    idx = sortperm(x)

    # PVA algorithm
    J = map(i->(Float64(y[i]*w[i]), w[i], [i]), idx)
    i = 1
    B₀ = J[i]
    while i < length(J)
        B₊ = J[i+1]
        if B₀[1] <= B₊[1] # step 1
            B₀ = B₊
            i += 1
        else # step 2
            ww = B₀[2] + B₊[2]
            J[i] = ((B₀[1]*B₀[2]+B₊[1]*B₊[2])/ww, ww, append!(B₀[3], B₊[3]))
            deleteat!(J, i+1)
            B₀ = J[i]
            while i > 1 # step 2.1
                B₋ = J[i-1]
                if B₀[1] <= B₋[1]
                    ww = B₀[2] + B₋[2]
                    J[i] = ((B₀[1]*B₀[2]+B₋[1]*B₋[2])/ww, ww, append!(B₀[3], B₋[3]))
                    deleteat!(J, i-1)
                    i-=1
                else
                    break
                end
            end
        end
    end
    [y for (y,w,ii) in J for i in sort(ii)]
end

