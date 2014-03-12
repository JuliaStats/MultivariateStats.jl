# Principal Component Analysis

#### PCA type

type PCA
    projection::Matrix{Float64}     # Projection matrix, size: (d, k)
    center::Vector{Float64}         # the center vector, length d or empty
    principalvars::Vector{Float64}  # variances along principal dimensions, length k
    residualvar::Float64            # variance of residue
    totalvar::Float64               # total variance of input samples, sum(prinvars) + residvar
    cumratios::Vector{Float64}      # cumulative ratio of variance preserved up to k-th dims

    function PCA(P::Matrix{Float64},        # projection matrix
                 center::Vector{Float64},   # center vector, length d or empty
                 pvars::Vector{Float64},    # variances along principal dimensions
                 tvar::Float64)             # total variance, i.e. sum(evals)

        if !isempty(center)
            size(P,1) == length(center) || 
                throw("Dimensions of projection matrix and center are inconsistent.")
        end
        k = size(P,2)
        length(pvars) == k || throw("pvars should have k values.")

        rvar = tvar - sum(pvars)
        new(P, center, pvars, rvar, tvar, cumsum(pvars) * inv(tvar))
    end
end

# methods on PCA

indim(m::PCA) = size(m.projection, 1)
outdim(m::PCA) = size(m.projection, 2)

show(io::IO, m::PCA) = print(io, "PCA (indim = $(indim(m)), outdim = $(outdim(m)))")

function dump(io::IO, m::PCA)
    show(io, m)
    println(io)
    println(io, "-----------------")
    print(io, "principal variances = ")
    printvecln(io, m.principalvars)
    print(io, "cumulative ratios   = ")
    printvecln(io, m.cumratios)
    println(io, "residual variance   =  $(m.residualvar)")
    println(io, "total variance      =  $(m.totalvar)")
    println(io, "projection:")
    printarrln(io, m.projection)
end

# transform and reconstruction

_centerize(m::PCA, x::DenseVector{Float64}) = 
    isempty(m.center) ? x : (x - m.center)

_centerize(m::PCA, x::DenseMatrix{Float64}) = 
    isempty(m.center) ? x : bsubtract(x, m.center, 1)

transform(m::PCA, x::DenseVecOrMat) = m.projection'_centerize(m, float64(x))

function reconstruct(m::PCA, y::DenseVector{Float64})
    x = m.projection * y
    if !isempty(m.center)
        add!(x, m.center)
    end
    return x
end

function reconstruct(m::PCA, y::DenseMatrix{Float64})
    x = m.projection * y
    if !isempty(m.center)
        badd!(x, m.center, 1)
    end
    return x
end

reconstruct(m::PCA, y::DenseVecOrMat) = reconstruct(m, float64(y))

#### PCA training

## internal (non-exported)

function pca_prepare(X::Matrix{Float64}, w::Vector{Float64}, center::Vector{Float64})
    # check sizes
    if !isempty(center)
        size(X,1) == length(center) || throw(DimensionMismatch("Inconsistent input dimensions."))
    end
    if !isempty(w)
        size(X,2) == length(w) || throw(DimensionMismatch("Incorrect length of the weight vector."))
    end

    if isempty(center)
        return isempty(w) ? X : scale(X, sqrt(w))
    else
        Z = bsubtract(X, center, 1)
        return isempty(w) ? Z : scale!(Z, sqrt(w))
    end
end

compute_cov(Z::Matrix{Float64}, tw::Float64) = BLAS.gemm('N', 'T', 1.0/tw, Z, Z)

function pca_decide_outdim(evals::Vector{Float64}, tvar::Float64,
                           maxoutdim::Int, ratio::Float64, ranktol::Float64)
    n = length(evals)

    # compute rank
    k::Int = 1
    if ratio < 1.0
        # by ratio of variance preserved in the principal subspace
        vthres = tvar * ratio
        v = evals[1]
        while k < n && v < vthres
            v += evals[k += 1]
        end
    else
        # by actual rank
        ethres = evals[1] * ranktol
        while k < n && evals[k+1] > ethres
            k += 1
        end
    end

    # cap at maxoutdim
    if maxoutdim > 0 && k > maxoutdim
        k = maxoutdim
    end
    return k
end

function makepca(U::Matrix{Float64}, 
                 evals::Vector{Float64}, 
                 center::Vector{Float64}, 
                 maxoutdim::Int, 
                 ratio::Float64, 
                 ranktol::Float64)

    si = sortperm(evals; rev=true)
    sevals = evals[si]
    tvar = sum(evals)

    k = pca_decide_outdim(sevals, tvar, maxoutdim, ratio, ranktol)
    pvars = sevals[1:k]
    P = U[:,si[1:k]]
    PCA(P, center, pvars, tvar)       
end

const default_ranktol=1.0e-12

## exported

function pcacov(C::Matrix{Float64}; 
                center::Vector{Float64}=Float64[], 
                maxoutdim::Int=-1, 
                ratio::Float64=1.0, 
                ranktol::Float64=default_ranktol)

    ef = eigfact(C)
    makepca(ef.vectors, ef.values, center, maxoutdim, ratio, ranktol)
end

function pcasvd(Z::Matrix{Float64}, tw::Float64;
                center::Vector{Float64}=Float64[], 
                maxoutdim::Int=-1, 
                ratio::Float64=1.0, 
                ranktol::Float64=default_ranktol)

    (U, svs, V) = svd(Z)
    makepca(U, scale!(abs2(svs), 1.0/tw), center, maxoutdim, ratio, ranktol)
end

function pca(X::Matrix{Float64};
             method::Symbol=:auto,
             weights::Vector{Float64}=Float64[], 
             zerocenter::Bool=false, 
             maxoutdim::Int=-1,
             ratio::Float64=1.0,
             ranktol::Float64=default_ranktol)

    # total weight
    tw = isempty(weights) ? float64(size(X,2)) : sum(weights)

    # compute center
    center::Vector{Float64}

    if zerocenter
        center = Float64[]
    else
        if isempty(weights)
            center = vec(mean(X, 2))
        else
            center = vec(scale!(wsum(weights, X, 2), 1.0 / tw))
        end
    end

    # prepare data that are centered & properly scaled
    Z = pca_prepare(X, weights, center)

    # dispatch according to method

    if method == :auto
        # choose a method based on size of X
        (d, n) = size(X)
        method = d < n ? :cov : :svd
    end

    if method == :cov
        C = compute_cov(Z, tw)
        return pcacov(C; center=center,
                      maxoutdim=maxoutdim, 
                      ratio=ratio,
                      ranktol=ranktol)
    elseif method == :svd
        return pcasvd(Z, tw; center=center, 
                      maxoutdim=maxoutdim, 
                      ratio=ratio,
                      ranktol=ranktol)
    else
        throw(ArgumentError("Invalid method $(method)"))
    end
end            

