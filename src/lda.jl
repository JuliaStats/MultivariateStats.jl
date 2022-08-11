# Linear Discriminant Analysis

#### Type to represent a linear discriminant functional

"""
A linear discriminant functional can be written as

```math
    f(\\mathbf{x}) = \\mathbf{w}^T \\mathbf{x} + b
```

Here, ``w`` is the coefficient vector, and ``b`` is the bias constant.
"""
struct LinearDiscriminant{T<:Real} <: RegressionModel
    w::Vector{T}
    b::T
end

#### function to solve linear discriminant
"""
    ldacov(C, μp, μn)

Performs LDA given a covariance matrix `C` and both mean vectors `μp` & `μn`.  Returns a linear discriminant functional of type [`LinearDiscriminant`](@ref).

*Parameters*
- `C`: The pooled covariane matrix (*i.e* ``(Cp + Cn)/2``)
- `μp`: The mean vector of the positive class.
- `μn`: The mean vector of the negative class.
"""
function ldacov(C::DenseMatrix{T},
                μp::DenseVector{T},
                μn::DenseVector{T}) where T<:Real

    w = cholesky(C) \ (μp - μn)
    ap = w ⋅ μp
    an = w ⋅ μn
    c = 2 / (ap - an)
    LinearDiscriminant(rmul!(w, c), 1 - c * ap)
end

"""
    ldacov(Cp, Cn, μp, μn)

Performs LDA given covariances and mean vectors. Returns a linear discriminant functional of type [`LinearDiscriminant`](@ref).

*Parameters*
- `Cp`: The covariance matrix of the positive class.
- `Cn`: The covariance matrix of the negative class.
- `μp`: The mean vector of the positive class.
- `μn`: The mean vector of the negative class.

**Note:** The coefficient vector is scaled such that ``w'μp + b = 1`` and ``w'μn + b = -1``.
"""
ldacov(Cp::DenseMatrix{T},
       Cn::DenseMatrix{T},
       μp::DenseVector{T},
       μn::DenseVector{T}) where T<:Real = ldacov(Cp + Cn, μp, μn)

"""
    evaluate(f, x::AbstractVector)

Evaluate the linear discriminant value, *i.e* ``w'x + b``, it returns a real value.
"""
evaluate(f::LinearDiscriminant, x::AbstractVector) = dot(f.w, x) + f.b

"""
    evaluate(f, X::AbstractMatrix)

Evaluate the linear discriminant value, *i.e* ``w'x + b``, for each sample in columns of `X`. The function returns a vector of length `size(X, 2)`.
"""
function evaluate(f::LinearDiscriminant, X::AbstractMatrix)
    R = transpose(X) * f.w
    if f.b != 0
        broadcast!(+, R, R, f.b)
    end
    return R
end

# RegressionModel interface

"""
    predict(f, x::AbstractVector)

Make prediction for the vector `x`. It returns `true` iff `evaluate(f, x)` is positive.
"""
predict(f::LinearDiscriminant, x::AbstractVector) = evaluate(f, x) > 0

"""
    predict(f, X::AbstractMatrix)

Make predictions for the matrix `X`.
"""
predict(f::LinearDiscriminant, X::AbstractMatrix) = Bool[y > 0 for y in evaluate(f, X)]

"""
    coef(f::LinearDiscriminant)

Return the coefficients of the linear discriminant model.
"""
coef(f::LinearDiscriminant) = (f.b, f.w)

"""
    coef(f::LinearDiscriminant)

Return the coefficients' names of the linear discriminant model.
"""
coefnames(f::LinearDiscriminant) = ["Bias", "Weights"]

"""
    dof(f::LinearDiscriminant)

Return the number of degrees of freedom in the linear discriminant model.
"""
dof(f::LinearDiscriminant) = length(f.w)+1

"""
    weights(f::LinearDiscriminant)

Return the linear discriminant model coefficient vector.
"""
weights(f::LinearDiscriminant) = f.w

"""
Get the length of the coefficient vector.
"""
length(f::LinearDiscriminant) = length(f.w)

"""
    fit(LinearDiscriminant, Xp, Xn; covestimator = SimpleCovariance())

Performs LDA given both positive and negative samples. The function accepts follwing parameters:

**Parameters**
- `Xp`: The sample matrix of the positive class.
- `Xn`: The sample matrix of the negative class.

**Keyword arguments:**
- `covestimator`: Custom covariance estimator for between-class covariance. The covariance matrix will be calculated as `cov(covestimator_between, #=data=#; dims=2, mean=zeros(#=...=#)`. Custom covariance estimators, available in other packages, may result in more robust discriminants for data with more features than observations.
"""
function fit(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T};
             covestimator::CovarianceEstimator = SimpleCovariance()) where T<:Real
    μp = vec(mean(Xp, dims=2))
    μn = vec(mean(Xn, dims=2))
    Zp = Xp .- μp
    Zn = Xn .- μn
    Cp = calcscattermat(covestimator, Zp)
    Cn = calcscattermat(covestimator, Zn)
    ldacov(Cp, Cn, μp, μn)
end

#==============================================================================#

#### Multiclass LDA Stats

"""
Resulting statistics of the multi-class LDA evaluation.
"""
mutable struct MulticlassLDAStats{T<:Real, M<:AbstractMatrix{T}, N<:AbstractMatrix{T}}
    dim::Int              # sample dimensions
    nclasses::Int         # number of classes
    cweights::Vector{T}   # class weights
    tweight::T            # total sample weight
    mean::Vector{T}       # overall sample mean
    cmeans::Matrix{T}     # class-specific means
    Sw::M                 # within-class scatter matrix
    Sb::N                 # between-class scatter matrix
end

mean(S::MulticlassLDAStats) = S.mean
classweights(S::MulticlassLDAStats) = S.cweights
classmeans(S::MulticlassLDAStats) = S.cmeans

withclass_scatter(S::MulticlassLDAStats) = S.Sw
betweenclass_scatter(S::MulticlassLDAStats) = S.Sb

function MulticlassLDAStats(cweights::Vector{T},
                            mean::Vector{T},
                            cmeans::Matrix{T},
                            Sw::AbstractMatrix{T},
                            Sb::AbstractMatrix{T}) where T<:Real
    d, nc = size(cmeans)
    length(mean) == d || throw(DimensionMismatch("Incorrect length of mean"))
    length(cweights) == nc || throw(DimensionMismatch("Incorrect length of cweights"))
    tw = sum(cweights)
    size(Sw) == (d, d) || throw(DimensionMismatch("Incorrect size of Sw"))
    size(Sb) == (d, d) || throw(DimensionMismatch("Incorrect size of Sb"))
    MulticlassLDAStats(d, nc, cweights, tw, mean, cmeans, Sw, Sb)
end

function multiclass_lda_stats(X::AbstractMatrix{T}, y::AbstractVector;
                              covestimator_within::CovarianceEstimator=SimpleCovariance(),
                              covestimator_between::CovarianceEstimator=SimpleCovariance()) where T<:Real
    # check sizes
    d = size(X, 1)
    n = size(X, 2)
    nc = length(unique(y))
    n ≥ nc || throw(ArgumentError("The number of samples is less than the number of classes"))
    length(y) == n || throw(DimensionMismatch("Inconsistent array sizes."))

    # compute class-specific weights and means
    cmeans, cweights, Z = center(X, y)

    Sw = calcscattermat(covestimator_within, Z)

    # compute between-class scattering
    mean = cmeans * Vector{T}(cweights ./ n)
    U = rmul!(cmeans .- mean, Diagonal(sqrt.(cweights)))
    Sb = calcscattermat(covestimator_between, U)

    return MulticlassLDAStats(Vector{T}(cweights), mean, cmeans, Sw, Sb)
end


#### Multiclass LDA

"""
A multi-class linear discriminant model type has following fields:
- `proj` is the projection matrix
- `pmeans` is the projected means of all classes
- `stats` is an instance of [`MulticlassLDAStats`](@ref) type that captures all statistics computed to train the model (which we will discuss later).
"""
mutable struct MulticlassLDA{T<:Real} <: RegressionModel
    proj::Matrix{T}
    pmeans::Matrix{T}
    stats::MulticlassLDAStats{T}
end

"""
    size(M::MulticlassLDA)

Get the input (*i.e* the dimension of the observation space) and output (*i.e* the dimension of the transformed features) dimensions of the model `M`.
"""
size(M::MulticlassLDA) = size(M.proj)
"""
    length(M::MulticlassLDA)

Get the sample dimensions.
"""
length(M::MulticlassLDA) = M.stats.dim
"""
    projection(M::MulticlassLDA)

Get the projection matrix (of size *d x p*).
"""
projection(M::MulticlassLDA) = M.proj
"""
    mean(M::MulticlassLDA)

Get the overall sample mean vector (of length *d*).
"""
mean(M::MulticlassLDA) = mean(M.stats)

"""
    classmeans(M)

Get the matrix comprised of class-specific means as columns (of size ``(d, m)``).
"""
classmeans(M::MulticlassLDA) = classmeans(M.stats)
"""
    classweights(M)

Get the weights of individual classes (a vector of length ``m``). If the samples are not weighted,
the weight equals the number of samples of each class.
"""
classweights(M::MulticlassLDA) = classweights(M.stats)
"""
withinclass_scatter(M)

    Get the within-class scatter matrix (of size ``(d, d)``).
"""
withclass_scatter(M::MulticlassLDA) = withclass_scatter(M.stats)
"""
betweenclass_scatter(M)

    Get the between-class scatter matrix (of size ``(d, d)``).
"""
betweenclass_scatter(M::MulticlassLDA) = betweenclass_scatter(M.stats)

"""
    predict(M::MulticlassLDA, x)

Transform input sample(s) in `x` to the output space of MC-LDA model `M`. Here, `x` can be either a sample vector or a matrix comprised of samples in columns.
"""
predict(M::MulticlassLDA, x::AbstractVecOrMat{T}) where {T<:Real} = M.proj'x

"""
    fit(MulticlassLDA, X, y; ...)

Perform multi-class LDA over a given data set `X` with corresponding labels `y`
with `nc` number of classes.

This function returns the resultant multi-class LDA model as an instance of [`MulticlassLDA`](@ref).

*Parameters*

- `X`:   the matrix of input samples, of size `(d, n)`. Each column in `X` is an observation.
- `y`:   the vector of class labels, of length `n`.

**Keyword arguments**

- `method`: The choice of methods:
    - `:gevd`: based on generalized eigenvalue decomposition (*default*).
    - `:whiten`: first derive a whitening transform from `Sw` and then solve the problem based on eigenvalue
    decomposition of the whiten `Sb`.
- `outdim`: The output dimension, i.e. dimension of the transformed space `min(d, nc-1)`
- `regcoef`: The regularization coefficient (*default:* `1.0e-6`). A positive value `regcoef * eigmax(Sw)`
    is added to the diagonal of `Sw` to improve numerical stability.
- `covestimator_between`: Custom covariance estimator for between-class covariance (*default:* `SimpleCovariance()`).
    The covariance matrix will be calculated as `cov(covestimator_between, #=data=#; dims=2, mean=zeros(#=...=#))`.
    Custom covariance estimators, available in other packages, may result in more robust discriminants for data
    with more features than observations.
- `covestimator_within`:  Custom covariance estimator for within-class covariance (*default:* `SimpleCovariance()`).
    The covariance matrix will be calculated as `cov(covestimator_within, #=data=#; dims=2, mean=zeros(nc))`.
    Custom covariance estimators, available in other packages, may result in more robust discriminants for data
    with more features than observations.

**Notes:**

The resultant projection matrix ``P`` satisfies:
```math
\\mathbf{P}^T (\\mathbf{S}_w + \\kappa \\mathbf{I}) \\mathbf{P} = \\mathbf{I}
```
Here, ``\\kappa`` equals `regcoef * eigmax(Sw)`. The columns of ``P`` are arranged in descending order of
the corresponding generalized eigenvalues.

Note that [`MulticlassLDA`](@ref) does not currently support the normalized version using ``\\mathbf{S}_w^*`` and
``\\mathbf{S}_b^*`` (see [`SubspaceLDA`](@ref)).
"""
function fit(::Type{MulticlassLDA}, X::AbstractMatrix{T}, y::AbstractVector;
             method::Symbol=:gevd,
             outdim::Int=min(size(X,1), length(unique(y))-1),
             regcoef::T=T(1.0e-6),
             covestimator_within::CovarianceEstimator=SimpleCovariance(),
             covestimator_between::CovarianceEstimator=SimpleCovariance()) where T<:Real

    multiclass_lda(multiclass_lda_stats(X, y;
                                        covestimator_within=covestimator_within,
                                        covestimator_between=covestimator_between);
                   method=method,
                   regcoef=regcoef,
                   outdim=outdim)
end

function multiclass_lda(S::MulticlassLDAStats{T};
                        method::Symbol=:gevd,
                        outdim::Int=min(S.dim, S.nclasses-1),
                        regcoef::T=T(1.0e-6)) where T<:Real

    P = mclda_solve(S.Sb, S.Sw, method, outdim, regcoef)
    MulticlassLDA(P, P'S.cmeans, S)
end

mclda_solve(Sb::AbstractMatrix{T}, Sw::AbstractMatrix{T}, method::Symbol, p::Int, regcoef::T) where T<:Real =
    mclda_solve!(copy(Sb), copy(Sw), method, p, regcoef)

function mclda_solve!(Sb::AbstractMatrix{T},
                      Sw::AbstractMatrix{T},
                      method::Symbol, p::Int, regcoef::T) where T<:Real

    p <= size(Sb, 1) || throw(ArgumentError("p cannot exceed sample dimension."))

    if method == :gevd
        regularize_symmat!(Sw, regcoef)
        E = eigen!(Symmetric(Sb), Symmetric(Sw))
        ord = sortperm(E.values; rev=true)
        P = E.vectors[:, ord[1:p]]

    elseif method == :whiten
        W = _lda_whitening!(Sw, regcoef)
        wSb = transpose(W) * (Sb * W)
        Eb = eigen!(Symmetric(wSb))
        ord = sortperm(Eb.values; rev=true)
        P = W * Eb.vectors[:, ord[1:p]]

    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    return P::Matrix{T}
end

function _lda_whitening!(C::AbstractMatrix{T}, regcoef::T) where T<:Real
    n = size(C,1)
    E = eigen!(Symmetric(C))
    v = E.values
    a = regcoef * maximum(v)
    for i = 1:n
        @inbounds v[i] = 1.0 / sqrt(v[i] + a)
    end
    return rmul!(E.vectors,  Diagonal(v))
end

#### SubspaceLDA

"""Subspace LDA model type has following fields:
- `projw`: the projection matrix of the subspace spanned by the between-class scatter
- `projLDA`: the projection matrix of the subspace spanned by the within-class scatter
- `λ`: the projection eigenvalues
- `cmeans`: the class centroids
- `cweights`: the class weights
"""
struct SubspaceLDA{T<:Real} <: RegressionModel
    projw::Matrix{T}
    projLDA::Matrix{T}
    λ::Vector{T}
    cmeans::Matrix{T}
    cweights::Vector{Int}
end

"""
    size(M)

Get the input (*i.e* the dimension of the observation space) and output (*i.e* the dimension of the subspace projection)
dimensions of the model `M`.
"""
size(M::SubspaceLDA) = (size(M.projw,1), size(M.projLDA, 2))
"""
    length(M)

Get dimension of the LDA model.
"""
length(M::SubspaceLDA) = size(M.projLDA, 2)
"""
    predict(M::SubspaceLDA, x)

Transform input sample(s) in `x` to the output space of LDA model `M`.
Here, `x` can be either a sample vector or a matrix comprised of samples in columns.
"""
predict(M::SubspaceLDA, x::AbstractVecOrMat{T}) where {T<:Real} = M.projLDA' * (M.projw' * x)
"""
    projection(M)

Get the projection matrix.
"""
projection(M::SubspaceLDA) = M.projw * M.projLDA

"""
    mean(M::SubspaceLDA)

Returns the mean vector of the subspace LDA model `M`.
"""
mean(M::SubspaceLDA) = vec(sum(M.cmeans * Diagonal(M.cweights / sum(M.cweights)), dims=2))

"""
    eigvals(M::SubspaceLDA)

Get the eigenvalues of the subspace LDA model `M`.
"""
eigvals(M::SubspaceLDA) = M.λ

classmeans(M::SubspaceLDA) = M.cmeans
classweights(M::SubspaceLDA) = M.cweights

"""
    fit(SubspaceLDA, X, y; normalize=true)

Fit an subspace projection of LDA model over a given data set `X` with corresponding
labels `y` using the equivalent of ``\\mathbf{S}_w^*`` and ``\\mathbf{S}_b^*```.

Note: Subspace LDA also supports the normalized version of LDA via the `normalize` keyword.
"""
function fit(::Type{SubspaceLDA}, X::AbstractMatrix{T},
             y::AbstractVector;
             normalize::Bool=false) where {T<:Real}
    d, n = size(X, 1), size(X, 2)
    nc = length(unique(y))
    n > nc || throw(ArgumentError("The number of samples is less than or equals the number of classes"))
    length(y) == n || throw(DimensionMismatch("Inconsistent array sizes."))
    # Compute centroids, class weights, and deviation from centroids
    # Note Sb = Hb*Hb', Sw = Hw*Hw'
    cmeans, cweights, Hw = center(X, y)
    dmeans = cmeans .- (normalize ? mean(cmeans, dims=2) : cmeans * (cweights / T(n)))
    Hb = normalize ? dmeans : dmeans * Diagonal(convert(Vector{T}, sqrt.(cweights)))
    if normalize
        Hw /= T(sqrt(n))
    end
    # Project to the subspace spanned by the within-class scatter
    # (essentially, PCA before LDA)
    Uw, Σw, _ = svd(Hw, full=false)
    keep = Σw .> sqrt(eps(T)) * maximum(Σw)
    count(keep) > 0 || throw(NullMatrixException("within-class covariance matrix is null"))
    projw = Uw[:,keep]
    pHb = projw' * Hb
    pHw = projw' * Hw
    λ, G = lda_gsvd(pHb, pHw, cweights)
    SubspaceLDA(projw, G, λ, cmeans, cweights)
end

struct NullMatrixException <: Exception
    msg::String
end

# Reference: Howland & Park (2006), "Generalizing discriminant analysis
# using the generalized singular value decomposition", IEEE
# Trans. Patt. Anal. & Mach. Int., 26: 995-1006.
function lda_gsvd(Hb::AbstractMatrix{T}, Hw::AbstractMatrix{T}, cweights::AbstractVector{Int}) where T<:Real
    nc = length(cweights)
    K = vcat(Hb', Hw') # `(n + nc) x m` matrix
    m = size(K, 2)
    P, R, Q = svd(K, full=false) # `Q` is `m x m` 
    tol = sqrt(eps(T))*maximum(R)
    t = count(x -> x > tol, R)
    R = R[1:t]
    Pk = P[1:nc, 1:t]
    U, ΣA, W = svd(Pk) # `W` is `t x t`
    ncnz = sum(cweights .> 0)
    # For the following codeblock `G` is `m x min(m, ncnz-1)`
    if t > ncnz-1
        G = Q[:,1:t]*(Diagonal(1 ./ R) * W[:,1:ncnz-1])
    else
        l = m <= ncnz - 1 ? m : ncnz - 1
        G = [Q[:,1:t]*(Diagonal(1 ./ R) * W) Q[:, t+1:l]]
    end
    # Normalize
    Gw = G' * Hw
    nrm = Gw * Gw'
    G = G ./ reshape(sqrt.(diag(nrm)), 1, size(G, 2))
    # Also get the eigenvalues
    Gw = G' * Hw
    Gb = G' * Hb
    λ = diag(Gb * Gb')./diag(Gw * Gw')
    λ, G
end

function center(X::AbstractMatrix{T}, y::AbstractVector) where T<:Real
    d, n = size(X,1), size(X,2)
    idxs = toindices(y)
    nc = maximum(idxs)
    # Calculate the class weights and means
    cmeans = zeros(T, d, nc)
    cweights = zeros(Int, nc)
    for j = 1:n
        k = idxs[j]
        for i = 1:d
            cmeans[i,k] += X[i,j]
        end
        cweights[k] += 1
    end
    for j = 1:nc
        cw = cweights[j]
        cw == 0 && continue
        for i = 1:d
            cmeans[i,j] /= cw
        end
    end
    # Compute differences from the means
    dX = Matrix{T}(undef, d, n)
    for j = 1:n
        k = idxs[j]
        for i = 1:d
            dX[i,j] = X[i,j] - cmeans[i,k]
        end
    end
    cmeans, cweights, dX
end
