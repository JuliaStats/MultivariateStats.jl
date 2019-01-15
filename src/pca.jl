# Principal Component Analysis

#### PCA type

struct PCA{T<:Real}
    mean     :: Vector{T} # sample mean: of length d (mean can be empty, which indicates zero mean)
    std      :: Vector{T} # sample standard deviation: length 0 indicates 1 std
    proj     :: Matrix{T} # projection matrix: of size d x p
    prinvars :: Vector{T} # principal variances: of length p
    tprinvar :: T         # total principal variance, i.e. sum(prinvars)
    tvar     :: T         # total input variance
end

## constructor

function PCA(
        mean  :: Vector{T},
        std   :: Vector{T},
        proj  :: Matrix{T},
        pvars :: Vector{T},
        tvar  :: T
    ) where {T<:Real}
    d, p = size(proj)
    (isempty(mean) || length(mean) == d) ||
        throw(DimensionMismatch("Dimensions of mean and proj are inconsistent."))
    (isempty(std) || length(std) == d) ||
        throw(DimensionMismatch("Dimensions of std and proj are inconsistent."))
    length(pvars) == p ||
        throw(DimensionMismatch("Dimensions of proj and pvars are inconsistent."))
    tpvar = sum(pvars)
    tpvar <= tvar || isapprox(tpvar,tvar) || throw(ArgumentError("principal variance cannot exceed total variance."))
    PCA(mean, std, proj, pvars, tpvar, tvar)
end

## properties

indim(M::PCA) = size(M.proj, 1)
outdim(M::PCA) = size(M.proj, 2)

mean(M::PCA) = fullmean(indim(M), M.mean)
std(M::PCA) = fullstd(indim(M), M.std)

projection(M::PCA) = M.proj

principalvar(M::PCA, i::Int) = M.prinvars[i]
principalvars(M::PCA) = M.prinvars

tprincipalvar(M::PCA) = M.tprinvar
tresidualvar(M::PCA) = M.tvar - M.tprinvar
tvar(M::PCA) = M.tvar

principalratio(M::PCA) = M.tprinvar / M.tvar

## use

transform(M::PCA{T}, x::AbstractVecOrMat{T}) where {T<:Real} =
    transpose(M.proj) * ztransform(x, M.mean, M.std)
reconstruct(M::PCA{T}, y::AbstractVecOrMat{T}) where {T<:Real} =
    deztransform(M.proj * y, M.mean, M.std)

## show & dump

function show(io::IO, M::PCA)
    print(io, "PCA(indim = $(indim(M)), outdim = $(outdim(M)), principalratio = $(principalratio(M)))")
end

function dump(io::IO, M::PCA)
    show(io, M)
    println(io)
    print(io, "principal vars: ")
    printvecln(io, M.prinvars)
    println(io, "total var = $(tvar(M))")
    println(io, "total principal var = $(tprincipalvar(M))")
    println(io, "total residual var  = $(tresidualvar(M))")
    println(io, "mean:")
    printvecln(io, mean(M))
    println(io, "projection:")
    printarrln(io, projection(M))
end


#### PCA Training

## auxiliary

const default_pca_pratio = 0.99

function check_pcaparams(d::Int, mean::AbstractVector, md::Int, pr::Real)
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Incorrect length of mean."))
    md >= 1 || error("maxoutdim must be a positive integer.")
    0.0 < pr <= 1.0 || throw(ArgumentError("pratio must be a positive real value with pratio ≤ 1.0."))
end

function check_pcaparams(d::Int, mean::AbstractVector, std::AbstractVector, md::Int, pr::Real)
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Incorrect length of mean."))
    isempty(std) || length(std) == d ||
        throw(DimensionMismatch("Incorrect length of std."))
    md >= 1 || error("maxoutdim must be a positive integer.")
    0.0 < pr <= 1.0 || throw(ArgumentError("pratio must be a positive real value with pratio ≤ 1.0."))
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

function pcacov(C::AbstractMatrix{T}, mean::Vector{T}, std::Vector{T};
                maxoutdim::Int=size(C,1),
                pratio::Real=default_pca_pratio) where {T<:Real}

    check_pcaparams(size(C,1), mean, std, maxoutdim, pratio)
    Eg = eigen(Symmetric(C))
    ev = Eg.values
    ord = sortperm(ev; rev=true)
    vsum = sum(ev)
    k = choose_pcadim(ev, ord, vsum, maxoutdim, pratio)
    v, P = extract_kv(Eg, ord, k)
    PCA(mean, std, P, v, vsum)
end

function pcasvd(Z::AbstractMatrix{T}, mean::Vector{T}, std::Vector{T}, tw::Real;
                maxoutdim::Int=min(size(Z)...),
                pratio::Real=default_pca_pratio) where {T<:Real}

    check_pcaparams(size(Z,1), mean, std, maxoutdim, pratio)
    Svd = svd(Z)
    v = Svd.S::Vector{T}
    U = Svd.U::Matrix{T}
    for i = 1:length(v)
        @inbounds v[i] = abs2(v[i]) / tw
    end
    ord = sortperm(v; rev=true)
    vsum = sum(v)
    k = choose_pcadim(v, ord, vsum, maxoutdim, pratio)
    si = ord[1:k]
    PCA(mean, std, U[:,si], v[si], vsum)
end

## interface functions

function fit(::Type{PCA}, X::AbstractMatrix{T};
             method::Symbol=:auto,
             maxoutdim::Int=size(X,1),
             pratio::Real=default_pca_pratio,
             mean=nothing, std=1) where {T<:Real}

    @assert !SparseArrays.issparse(X) "Use Kernel PCA for sparce arrays"

    d, n = size(X)

    # choose method
    if method == :auto
        method = d < n ? :cov : :svd
    end

    # process mean
    mv = preprocess_mean(X, mean)
    sv = preprocess_std(X, std, isempty(mv) ? zeros(T, d) : mv)

    # delegate to core
    if method == :cov
        if isempty(sv) && isempty(mv)
            C = cov2cor(covm(X, 0, 2), ones(T, d))
        elseif isempty(mv)
            C = cov2cor(covm(X, 0, 2), sv)
        elseif isempty(sv)
            C = covm(X, mv, 2)
        else
            C = cov2cor(covm(X, mv, 2), sv)
        end
        M = pcacov(C, mv, sv; maxoutdim=maxoutdim, pratio=pratio)
    elseif method == :svd
        Z = ztransform(X, mv, sv)
        M = pcasvd(Z, mv, sv, n; maxoutdim=maxoutdim, pratio=pratio)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::PCA
end
