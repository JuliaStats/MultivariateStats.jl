# Principal Component Analysis

"""
Linear Principal Component Analysis
"""
struct PCA{T<:Real} <: LinearDimensionalityReduction
    mean::AbstractVector{T}     # sample mean: of length d (mean can be empty, which indicates zero mean)
    proj::AbstractMatrix{T}     # projection matrix: of size d x p
    prinvars::AbstractVector{T} # principal variances: of length p
    tprinvar::T                 # total principal variance, i.e. sum(prinvars)
    tvar::T                     # total input variance
end

## constructor

function PCA(mean::AbstractVector{T}, proj::AbstractMatrix{T}, pvars::AbstractVector{T}, tvar::T) where {T<:Real}
    d, p = size(proj)
    (isempty(mean) || length(mean) == d) ||
        throw(DimensionMismatch("Dimensions of mean and projection matrix are inconsistent."))
    length(pvars) == p ||
        throw(DimensionMismatch("Dimensions of projection matrix and principal variables are inconsistent."))
    tpvar = sum(pvars)
    tpvar <= tvar || isapprox(tpvar,tvar) || throw(ArgumentError("principal variance cannot exceed total variance."))
    PCA(mean, proj, pvars, tpvar, tvar)
end

## properties
"""
    size(M)

Returns a tuple with the dimensions of input (the dimension of the observation space)
and output (the dimension of the principal subspace).
"""
size(M::PCA) = size(M.proj)

"""
    mean(M::PCA)

Returns the mean vector (of length `d`).
"""
mean(M::PCA) = fullmean(size(M.proj,1), M.mean)

"""
    projection(M::PCA)

Returns the projection matrix (of size `(d, p)`). Each column of the projection matrix corresponds to a principal component.
The principal components are arranged in descending order of the corresponding variances.
"""
projection(M::PCA) = M.proj

"""
    eigvecs(M::PCA)

Get the eigenvalues of the PCA model `M`.
"""
eigvecs(M::PCA) = projection(M)

"""
    principalvars(M::PCA)

Returns the variances of principal components.
"""
principalvars(M::PCA) = M.prinvars
principalvar(M::PCA, i::Int) = M.prinvars[i]

"""
    eigvals(M::PCA)

Get the eigenvalues of the PCA model `M`.
"""
eigvals(M::PCA) = principalvars(M)

"""
    tprincipalvar(M::PCA)

Returns the total variance of principal components, which is equal to `sum(principalvars(M))`.
"""
tprincipalvar(M::PCA) = M.tprinvar

"""
    tresidualvar(M::PCA)

Returns the total residual variance.
"""
tresidualvar(M::PCA) = M.tvar - M.tprinvar

"""
    var(M::PCA)

Returns the total observation variance, which is equal to `tprincipalvar(M) + tresidualvar(M)`.
"""
var(M::PCA) = M.tvar

"""
    r2(M::PCA)
    principalratio(M::PCA)

Returns the ratio of variance preserved in the principal subspace, which is equal to `tprincipalvar(M) / var(M)`.
"""
r2(M::PCA) = M.tprinvar / M.tvar
const principalratio = r2

"""
    loadings(M::PCA)

Returns model loadings, i.e. the weights for each original variable when calculating the principal component.
"""
loadings(M::PCA) = sqrt.(principalvars(M))' .* projection(M)

## use

"""
    predict(M::PCA, x::AbstractVecOrMat{<:Real})

Given a PCA model `M`, retur transform observations `x` into principal components space, as

\$\\mathbf{y} = \\mathbf{P}^T (\\mathbf{x} - \\boldsymbol{\\mu})\$

Here, `x` can be either a vector of length `d` or a matrix where each column is an observation,
and `\\mathbf{P}` is the projection matrix.
"""
predict(M::PCA, x::AbstractVecOrMat{T}) where {T<:Real} = transpose(M.proj) * centralize(x, M.mean)

"""
    reconstruct(M::PCA, y::AbstractVecOrMat{<:Real})

Given a PCA model `M`, returns a (approximately) reconstructed observations
from principal components space, as

\$\\tilde{\\mathbf{x}} = \\mathbf{P} \\mathbf{y} + \\boldsymbol{\\mu}\$

Here, `y` can be either a vector of length `p` or a matrix where each column
gives the principal components for an observation, and \$\\mathbf{P}\$ is the projection matrix.
"""
reconstruct(M::PCA, y::AbstractVecOrMat{T}) where {T<:Real} = decentralize(M.proj * y, M.mean)

## show & dump
function show(io::IO, M::PCA)
    idim, odim = size(M)
    print(io, "PCA(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
end

function show(io::IO, ::MIME"text/plain", M::PCA)
    idim, odim = size(M)
    print(io, "PCA(indim = $idim, outdim = $odim, principalratio = $(r2(M)))")
    ldgs = loadings(M)
    rot = diag(ldgs' * ldgs)
    ldgs = ldgs[:, sortperm(rot, rev=true)]
    ldgs_signs = sign.(sum(ldgs, dims=1))
    replace!(ldgs_signs, 0=>1)
    ldgs = ldgs * diagm(0 => ldgs_signs[:])
    print(io, "\n\nPattern matrix (unstandardized loadings):\n")
    cft = CoefTable(ldgs, string.("PC", 1:odim), string.("", 1:idim))
    print(io, cft)
    print(io, "\n\n")
    print(io, "Importance of components:\n")
    λ = eigvals(M)
    prp = λ ./ var(M)
    prpv = λ ./ sum(λ)
    names = ["SS Loadings (Eigenvalues)",
             "Variance explained", "Cumulative variance",
             "Proportion explained", "Cumulative proportion"]
    cft = CoefTable(vcat(λ', prp', cumsum(prp)',  prpv', cumsum(prpv)'),
                    string.("PC", 1:odim), names)
    print(io, cft)
end

#### PCA Training

## auxiliary

const default_pca_pratio = 0.99

function check_pcaparams(d::Int, mean::AbstractVector, md::Int, pr::Real)
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Incorrect length of mean."))
    md >= 1 || error("`maxoutdim` parameter must be a positive integer.")
    0.0 < pr <= 1.0 || throw(ArgumentError("principal ratio must be a positive real value ≤ 1.0."))
end

function choose_pcadim(v::AbstractVector{T}, ord::Vector{Int}, vsum::T, md::Int,
                       pr::Real) where {T<:Real}
    md = min(length(v), md)
    k = 1
    a = v[ord[1]]
    thres = vsum * convert(T, pr)
    while k < md && a < thres
        a += v[ord[k += 1]]
    end
    return k
end


## core algorithms
"""
    pcacov(C, mean; ...)

Compute and return a PCA model based on eigenvalue decomposition of a given covariance matrix `C`.

**Parameters:**
- `C`: The covariance matrix of the samples.
- `mean`: The mean vector of original samples, which can be a vector of length `d`,
           or an empty vector `Float64[]` indicating a zero mean.

*Note:* This function accepts two keyword arguments: `maxoutdim` and `pratio`.
"""
function pcacov(C::AbstractMatrix{T}, mean::AbstractVector{T};
                maxoutdim::Int=size(C,1),
                pratio::Real=default_pca_pratio) where {T<:Real}

    check_pcaparams(size(C,1), mean, maxoutdim, pratio)
    Eg = eigen(Symmetric(C))
    ev = Eg.values
    ord = sortperm(ev; rev=true)
    vsum = sum(ev)
    k = choose_pcadim(ev, ord, vsum, maxoutdim, pratio)
    v, P = extract_kv(Eg, ord, k)
    PCA(mean, P, v, vsum)
end

"""
    pcasvd(Z, mean, tw; ...)

Compute and return a PCA model based on singular value decomposition of a centralized sample matrix `Z`.

**Parameters:**
- `Z`: a matrix of centralized samples.
- `mean`: The mean vector of the **original** samples, which can be a vector of length `d`,
          or an empty vector `Float64[]` indicating a zero mean.
- `n`: a number of samples.

*Note:* This function accepts two keyword arguments: `maxoutdim` and `pratio`.
"""
function pcasvd(Z::AbstractMatrix{T}, mean::AbstractVector{T}, n::Real;
                maxoutdim::Int=min(size(Z)...),
                pratio::Real=default_pca_pratio) where {T<:Real}

    check_pcaparams(size(Z,1), mean, maxoutdim, pratio)
    Svd = svd(Z)
    v = Svd.S::Vector{T}
    U = Svd.U::Matrix{T}
    for i = 1:length(v)
        @inbounds v[i] = abs2(v[i]) / n
    end
    ord = sortperm(v; rev=true)
    vsum = sum(v)
    k = choose_pcadim(v, ord, vsum, maxoutdim, pratio)
    si = ord[1:k]
    PCA(mean, U[:,si], v[si], vsum)
end

## interface functions
"""
    fit(PCA, X; ...)

Perform PCA over the data given in a matrix `X`. Each column of `X` is an **observation**.

**Keyword arguments**

- `method`: The choice of methods:
    - `:auto`: use `:cov` when `d < n` or `:svd` otherwise (*default*).
    - `:cov`: based on covariance matrix decomposition.
    - `:svd`: based on SVD of the input data.
- `maxoutdim`: The output dimension, i.e. dimension of the transformed space (*min(d, nc-1)*)
- `pratio`: The ratio of variances preserved in the principal subspace (*0.99*)
- `mean`: The mean vector, which can be either of
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (*default*)
    - a pre-computed mean vector

**Notes:**

- The output dimension `p` depends on both `maxoutdim` and `pratio`, as follows. Suppose
  the first `k` principal components preserve at least `pratio` of the total variance, while the
  first `k-1` preserves less than `pratio`, then the actual output dimension will be \$\\min(k, maxoutdim)\$.

- This function calls [`pcacov`](@ref) or [`pcasvd`](@ref) internally, depending on the choice of method.
"""
function fit(::Type{PCA}, X::AbstractMatrix{T};
             method::Symbol=:auto,
             maxoutdim::Int=size(X,1),
             pratio::Real=default_pca_pratio,
             mean=nothing) where {T<:Real}

    @assert !SparseArrays.issparse(X) "Use Kernel PCA for sparse arrays"

    d, n = size(X)

    # choose method
    if method == :auto
        method = d < n ? :cov : :svd
    end

    # process mean
    mv = preprocess_mean(X, mean)

    # delegate to core
    if method == :cov
        C = covm(X, isempty(mv) ? 0 : mv, 2)
        M = pcacov(C, mv; maxoutdim=maxoutdim, pratio=pratio)
    elseif method == :svd
        Z = centralize(X, mv)
        M = pcasvd(Z, mv, n; maxoutdim=maxoutdim, pratio=pratio)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::PCA
end
