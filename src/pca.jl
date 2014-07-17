# Principal Component Analysis

#### PCA type

type PCA
    mean::Vector{Float64}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    proj::Matrix{Float64}       # projection matrix: of size d x p
    prinvars::Vector{Float64}   # principal variances: of length p
    tprinvar::Float64           # total principal variance, i.e. sum(prinvars)
    tvar::Float64               # total input variance
end          

## constructor

function PCA(mean::Vector{Float64}, proj::Matrix{Float64}, pvars::Vector{Float64}, tvar::Float64)
    d, p = size(proj)
    (isempty(mean) || length(mean) == d) ||
        throw(DimensionMismatch("Dimensions of mean and proj are inconsistent."))
    length(pvars) == p ||
        throw(DimensionMismatch("Dimensions of proj and pvars are inconsistent."))
    tpvar = sum(pvars)
    tpvar <= tvar || error("principal variance cannot exceed total variance.")
    PCA(mean, proj, pvars, tpvar, tvar)
end

## properties

indim(M::PCA) = size(M.proj, 1)
outdim(M::PCA) = size(M.proj, 2)

Base.mean(M::PCA) = fullmean(indim(M), M.mean)

projection(M::PCA) = M.proj

principalvar(M::PCA, i::Integer) = M.prinvars[i]
principalvars(M::PCA) = M.prinvars

tprincipalvar(M::PCA) = M.tprinvar
tresidualvar(M::PCA) = M.tvar - M.tprinvar
tvar(M::PCA) = M.tvar

principalratio(M::PCA) = M.tprinvar / M.tvar

## use

transform{T<:Real}(M::PCA, x::AbstractVecOrMat{T}) = At_mul_B(M.proj, centralize(x, M.mean))
reconstruct{T<:Real}(M::PCA, y::AbstractVecOrMat{T}) = decentralize(M.proj * y, M.mean)

## show & dump

function show(io::IO, M::PCA)
    pr = @sprintf("%.5f", principalratio(M))
    print(io, "PCA(indim = $(indim(M)), outdim = $(outdim(M)), principalratio = $pr)")
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

function check_pcaparams(d::Int, mean::Vector, md::Int, pr::Float64)
    isempty(mean) || length(mean) == d ||
        throw(DimensionMismatch("Incorrect length of mean."))
    md >= 1 || error("maxoutdim must be a positive integer.")
    0.0 < pr <= 1.0 || error("pratio must be a positive real value with pratio <= 1.0.")
end


function choose_pcadim(v::AbstractVector, ord::Vector{Int}, vsum::Float64, md::Int, pr::Float64)
    md = min(length(v), md)
    k = 1
    a = v[ord[1]]
    thres = vsum * pr
    while k < md && a < thres
        a += v[ord[k += 1]]
    end
    return k
end


## core algorithms

function pcacov(C::Matrix{Float64}, mean::Vector{Float64}; 
                maxoutdim::Int=size(C,1), 
                pratio::Float64=default_pca_pratio)

    check_pcaparams(size(C,1), mean, maxoutdim, pratio)
    Eg = eigfact!(Symmetric(copy(C)))
    ev = Eg.values
    ord = sortperm(ev; rev=true)
    vsum = sum(ev)
    k = choose_pcadim(ev, ord, vsum, maxoutdim, pratio)
    v, P = extract_kv(Eg, ord, k)
    PCA(mean, P, v, vsum)
end

function pcastd(Z::Matrix{Float64}, mean::Vector{Float64}, tw::Real; 
                maxoutdim::Int=min(size(Z)...),
                pratio::Float64=default_pca_pratio)

    check_pcaparams(size(Z,1), mean, maxoutdim, pratio)
    Svd = svdfact(Z)
    v = Svd.S::Vector{Float64}
    U = Svd.U::Matrix{Float64}
    for i = 1:length(v)
        @inbounds v[i] = abs2(v[i]) / tw
    end
    ord = sortperm(v; rev=true)
    vsum = sum(v)
    k = choose_pcadim(v, ord, vsum, maxoutdim, pratio)
    si = ord[1:k]
    PCA(mean, U[:,si], v[si], vsum)
end

## interface functions

function fit(::Type{PCA}, X::Matrix{Float64}; 
             method::Symbol=:auto, 
             maxoutdim::Int=size(X,1), 
             pratio::Float64=default_pca_pratio, 
             mean=nothing)

    d, n = size(X)
    
    # choose method
    if method == :auto
        method = d < n ? :cov : :std
    end

    # process mean
    mv = (mean == nothing ? vec(Base.mean(X, 2)) :
          mean == 0 ? Float64[] : mean)::Vector{Float64}

    # delegate to core
    if method == :cov
        C = cov(X; vardim=2, mean=isempty(mv) ? 0 : mv)::Matrix{Float64}
        M = pcacov(C, mv; maxoutdim=maxoutdim, pratio=pratio)
    elseif method == :std
        Z = centralize(X, mv)
        M = pcastd(Z, mv, float64(n); maxoutdim=maxoutdim, pratio=pratio)
    else
        error("Invalid method name $(method)")
    end

    return M::PCA
end
