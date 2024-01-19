# Canonical Correlation Analysis

#### CCA Type
"""
Canonical Correlation Analysis Model
"""
struct CCA{T<:Real} <: RegressionModel
    xmean::Vector{T}  # sample mean of X: of length dx (can be empty)
    ymean::Vector{T}  # sample mean of Y: of length dy (can be empty)
    xproj::Matrix{T}  # projection matrix for X, of size (dx, p)
    yproj::Matrix{T}  # projection matrix for Y, of size (dy, p)
    corrs::Vector{T}  # correlations, of length p

    function CCA(xm::Vector{T},
                 ym::Vector{T},
                 xp::Matrix{T},
                 yp::Matrix{T},
                 crs::Vector{T}) where T<:Real

        dx, px = size(xp)
        dy, py = size(yp)

        isempty(xm) || length(xm) == dx ||
            throw(DimensionMismatch("Incorrect length of xmean."))

        isempty(ym) || length(ym) == dy ||
            throw(DimensionMismatch("Incorrect length of ymean."))

        px == py ||
            throw(DimensionMismatch("xproj and yproj should have the same number of columns."))

        length(crs) == px ||
            throw(DimensionMismatch("Incorrect length of corrs."))

        new{T}(xm, ym, xp, yp, crs)
    end
end

## properties

"""
    size(M:CCA)

Return a tuple with the dimension of `X`, `Y`, and the output dimension.
"""
size(M::CCA) = (size(M.xproj, 1), size(M.yproj, 1), size(M.xproj, 2))

"""
    mean(M::CCA, c::Symbol)

Get the mean vector for the component `c` of the model `M`.
The component parameter can be `:x` or `:y`.
"""
function mean(M::CCA, c::Symbol)
    xi, yi, o = size(M)
    if c == :x
        fullmean(xi, M.xmean)
    elseif c == :y
        fullmean(yi, M.ymean)
    else
        throw(ArgumentError("Unknown component $c"))
    end
end

"""
    projection(M::CCA, c::Symbol)

Get the projection matrix for the component `c` of the model `M`.
The component parameter can be `:x` or `:y`.
"""
function projection(M::CCA, c::Symbol)
    if c == :x
        M.xproj
    elseif c == :y
        M.yproj
    else
        throw(ArgumentError("Unknown component $c"))
    end
end

"""
    cor(M::CCA)

The correlations of the projected components (a vector of length `p`).
"""
cor(M::CCA) = M.corrs

## use
"""
    predict(M::CCA, Z::AbstractVecOrMat{<:Real}, c::Symbol)

Given a [`CCA`](@ref) model, one can transform observations into both spaces into a common space, as

```math
\\mathbf{z}_x = \\mathbf{P}_x^T (\\mathbf{x} - \\boldsymbol{\\mu}_x) \\\\
\\mathbf{z}_y = \\mathbf{P}_y^T (\\mathbf{y} - \\boldsymbol{\\mu}_y)
```

Here, ``\\mathbf{P}_x`` and ``\\mathbf{P}_y`` are projection matrices for ``X`` and ``Y``;
``\\boldsymbol{\\mu}_x`` and ``\\boldsymbol{\\mu}_y`` are mean vectors.

Parameter `Z` can be either a vector of length `dx`, `dy`, or a matrix where each column is an observation. The component parameter `c` can be `:x` or `:y`.
"""
function predict(M::CCA, Z::AbstractVecOrMat{<:Real}, c::Symbol)
    if c == :x
        transpose(M.xproj) * centralize(Z, M.xmean)
    elseif c == :y
        transpose(M.yproj) * centralize(Z, M.ymean)
    else
        throw(ArgumentError("Unknown component $c"))
    end
end

## show

function show(io::IO, M::CCA)
    xi, yi, o = size(M)
    print(io, "CCA (xindim = $xi, yindim = $yi, outdim = $o)")
end


#### Perform CCA on data

"""
    ccacov(Cxx, Cyy, Cxy, xmean, ymean, p)

Compute CCA based on analysis of the given covariance matrices, using generalized
eigenvalue decomposition, and return [`CCA`](@ref) model.

Parameters:
- `Cxx`: The covariance matrix of `X`.
- `Cyy`: The covariance matrix of `Y`.
- `Cxy`: The covariance matrix between `X` and `Y`.
- `xmean`: The mean vector of the **original** samples of `X`, which can be
a vector of length `dx`, or an empty vector indicating a zero mean.
- `ymean`: The mean vector of the **original** samples of `Y`, which can be
a vector of length `dy`, or an empty vector indicating a zero mean.
- `p`: The output dimension, *i.e* the dimension of the common space.
"""
function ccacov(Cxx::DenseMatrix{T},
                Cyy::DenseMatrix{T},
                Cxy::DenseMatrix{T},
                xmean::Vector{T},
                ymean::Vector{T},
                p::Int) where T<:Real

    # argument checking
    dx, dx2 = size(Cxx)
    dy, dy2 = size(Cyy)
    dx == dx2 || error("Cxx must be a square matrix.")
    dy == dy2 || error("Cyy must be a square matrix.")
    size(Cxy) == (dx, dy) ||
        throw(DimensionMismatch("size(Cxy) should be equal to (dx, dy)"))

    isempty(xmean) || length(xmean) == dx ||
        throw(DimensionMismatch("Incorrect length of xmean."))

    isempty(ymean) || length(ymean) == dy ||
        throw(DimensionMismatch("Incorrect length of ymean."))

    1 <= p <= min(dx, dy) ||
        throw(DimensionMismatch(""))

    _ccacov(Cxx, Cyy, Cxy, xmean, ymean, p)
end

function _ccacov(Cxx, Cyy, Cxy, xmean, ymean, p::Int)
    dx = size(Cxx, 1)
    dy = size(Cyy, 1)

    # solve Px and Py

    if dx <= dy
        # solve Px: (Cxy * inv(Cyy) * Cyx) Px = λ Cxx * Px
        # compute Py: inv(Cyy) * Cyx * Px

        G = cholesky(Cyy) \ Cxy'
        Ex = eigen(Symmetric(Cxy * G), Symmetric(Cxx))
        ord = sortperm(Ex.values; rev=true)
        vx, Px = extract_kv(Ex, ord, p)
        Py = qnormalize!(G * Px, Cyy)
    else
        # solve Py: (Cyx * inv(Cxx) * Cxy) Py = λ Cyy Py
        # compute Px: inv(Cx) * Cxy * Py

        H = cholesky(Cxx) \ Cxy
        Ey = eigen(Symmetric(Cxy'H), Symmetric(Cyy))
        ord = sortperm(Ey.values; rev=true)
        vy, Py = extract_kv(Ey, ord, p)
        Px = qnormalize!(H * Py, Cxx)
    end

    # compute correlations
    # Note: Px' * Cxx * Px == I
    #       Py' * Cyy * Py == I
    crs = coldot(Px, Cxy * Py)

    # construct CCA model
    CCA(xmean, ymean, Px, Py, crs)
end

"""
    ccasvd(Zx, Zy, xmean, ymean, p)

Compute CCA based on singular value decomposition of centralized sample matrices `Zx` and `Zy`, and return [`CCA`](@ref) model[^1].

Parameters:
- `Zx`: The centralized sample matrix for `X`.
- `Zy`: The centralized sample matrix for `Y`.
- `xmean`: The mean vector of the **original** samples of `X`, which can be
a vector of length `dx`, or an empty vector indicating a zero mean.
- `ymean`: The mean vector of the **original** samples of `Y`, which can be
a vector of length `dy`, or an empty vector indicating a zero mean.
- `p`: The output dimension, *i.e* the dimension of the common space.
"""
function ccasvd(Zx::DenseMatrix{T},
                Zy::DenseMatrix{T},
                xmean::Vector{T},
                ymean::Vector{T},
                p::Int) where T<:Real

    dx, n = size(Zx)
    dy, n2 = size(Zy)
    n == n2 ||
        throw(DimensionMismatch("Zx and Zy must have the same number of columns."))

    isempty(xmean) || length(xmean) == dx ||
        throw(DimensionMismatch("Incorrect length of xmean."))

    isempty(ymean) || length(ymean) == dy ||
        throw(DimensionMismatch("Incorrect length of ymean."))

    1 <= p <= min(dx, dy) ||
        throw(DimensionMismatch(""))

    _ccasvd(Zx, Zy, xmean, ymean, p)
end

# The implementation is partly based on:
#
#   David Weenink.
#   Canonical Correlation Analysis.
#   Institute of Phonetic Sciences, Univ. of Amsterdam,
#   Proceedings 25 (2003), 81-99.
#
#   Note: in this paper, each row is considered as an observation.
#   The algorithm is adapted to the column-major format here.
#
function _ccasvd(Zx::DenseMatrix{T}, Zy::DenseMatrix{T}, xmean::Vector{T}, ymean::Vector{T}, p::Int) where T<:Real
    # svd factorization of Z

    n = size(Zx, 2)

    # svd decomposition
    Sx = svd(Zx)
    Sy = svd(Zy)
    S = svd!(Sx.Vt * transpose(Sy.Vt)) # svd of Vx * Vy'

    # compute Px and Py
    ord = sortperm(S.S; rev=true)
    si = ord[1:p]
    Px = rmul!(Sx.U, Diagonal(1.0 ./ Sx.S)) * S.U[:, si]
    Py = rmul!(Sy.U, Diagonal(1.0 ./ Sy.S)) * S.V[:, si]

    # scale so that Px' * Cxx * Py == I
    #           and Py' * Cyy * Py == I,
    #
    # with Cxx = Zx * Zx' / (n - 1)
    #      Cyy = Zy * Zy' / (n - 1)
    #
    rmul!(Px, sqrt(n-1))
    rmul!(Py, sqrt(n-1))

    # compute correlations
    crs = rmul!(coldot(Zx'Px, Zy'Py), one(T)/(n-1))

    # construct CCA model
    CCA(xmean, ymean, Px, Py, crs)
end

## interface functions

"""
    fit(CCA, X, Y; ...)

Perform CCA over the data given in matrices `X` and `Y`.
Each column of `X` and `Y` is an observation.

`X` and `Y` should have the same number of columns (denoted by `n` below).

This method returns an instance of [`CCA`](@ref).

**Keyword arguments:**
- `method`: The choice of methods:
    - `:cov`: based on covariance matrices
    - `:svd`: based on SVD of the input data (*default*)
- `outdim`: The output dimension, *i.e* dimension of the common space (*default*: `min(dx, dy, n)`)
- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (*default*)
    - a pre-computed mean vector

**Notes:** This function calls [`ccacov`](@ref) or [`ccasvd`](@ref) internally, depending on the choice of method.
"""
function fit(::Type{CCA}, X::AbstractMatrix{T}, Y::AbstractMatrix{T};
             outdim::Int=min(min(size(X)...), min(size(Y)...)),
             method::Symbol=:svd,
             xmean=nothing,
             ymean=nothing) where T<:Real

    dx, n = size(X)
    dy, n2 = size(Y)

    n2 == n ||
        throw(DimensionMismatch("X and Y should have the same number of columns."))

    (n >= dx && n >= dy) ||
        @warn("CCA would be numerically unstable when n < dx or n < dy.")

    xmv = preprocess_mean(X, xmean)
    ymv = preprocess_mean(Y, ymean)

    Zx = centralize(X, xmv)
    Zy = centralize(Y, ymv)

    if method == :cov
        Cxx = rmul!(Zx*transpose(Zx), inv(n - 1))
        Cyy = rmul!(Zy*transpose(Zy), inv(n - 1))
        Cxy = rmul!(Zx*transpose(Zy), inv(n - 1))
        M = ccacov(Cxx, Cyy, Cxy, xmv, ymv, outdim)
    elseif method == :svd
        M = ccasvd(Zx, Zy, xmv, ymv, outdim)
    else
        error("Invalid method name $(method)")
    end

    return M::CCA
end
