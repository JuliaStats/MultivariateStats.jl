# Principal Component Analysis

#### PCA type

type PCA
    mean::Vector{Float64}       # of length d (mean can be empty, which indicates zero mean)
    proj::Matrix{Float64}       # projection matrix: of size d x p
    prinvars::Vector{Float64}   # principal variances: of length p
    tprinvar::Float64           # total principal variance, i.e. sum(prinvars)
    tresivar::Float64           # total variance of residual
end          

# constructor

function PCA(mean::Vector{Float64}, proj::Matrix{Float64}, pvars::Vector{Float64}, trvar::Float64)
    d, p = size(proj)
    (isempty(mean) || length(mean) == d) ||
        throw(DimensionMismatch("Dimensions of mean and proj are inconsistent."))
    length(pvars) == p ||
        throw(DimensionMismatch("Dimensions of proj and pvars are inconsistent."))
    PCA(mean, proj, pvars, sum(pvars), trvar)
end

# properties

indim(M::PCA) = size(M.proj, 1)
outdim(M::PCA) = size(M.proj, 2)

Base.mean(M::PCA) = (isempty(M.mean) ? zeros(indim(M)) : M.mean)::Vector{Float64}

projection(M::PCA) = M.proj

principalvar(M::PCA, i::Integer) = M.prinvars[i]
principalvars(M::PCA) = M.prinvars

tprincipalvar(M::PCA) = M.tprinvar
tresidualvar(M::PCA) = M.tresivar

principalratio(M::PCA) = M.tprinvar / (M.tprinvar + M.tresivar)

# use

transform{T<:Real}(M::PCA, x::AbstractVecOrMat{T}) = At_mul_B(M.proj, centralize(x, M.mean))
reconstruct{T<:Real}(M::PCA, y::AbstractVecOrMat{T}) = decentralize(M.proj * y, M.mean)

