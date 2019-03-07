# Linear Discriminant Analysis

#### Type to represent a linear discriminant functional

abstract type Discriminant{T} end

struct LinearDiscriminant{T<:Real} <: Discriminant{T}
    w::Vector{T}
    b::T
end

length(f::LinearDiscriminant) = length(f.w)

evaluate(f::LinearDiscriminant, x::AbstractVector) = dot(f.w, x) + f.b

function evaluate(f::LinearDiscriminant, X::AbstractMatrix)
    R = transpose(X) * f.w
    if f.b != 0
        broadcast!(+, R, R, f.b)
    end
    return R
end

predict(f::Discriminant, x::AbstractVector) = evaluate(f, x) > 0

predict(f::Discriminant, X::AbstractMatrix) = (Y = evaluate(f, X); Bool[y > 0 for y in Y])


#### function to solve linear discriminant

function ldacov(C::DenseMatrix{T},
                μp::DenseVector{T},
                μn::DenseVector{T}) where T<:Real

    w = cholesky(C) \ (μp - μn)
    ap = w ⋅ μp
    an = w ⋅ μn
    c = 2 / (ap - an)
    LinearDiscriminant(rmul!(w, c), 1 - c * ap)
end

ldacov(Cp::DenseMatrix{T},
       Cn::DenseMatrix{T},
       μp::DenseVector{T},
       μn::DenseVector{T}) where T<:Real = ldacov(Cp + Cn, μp, μn)

#### interface functions

function fit(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T};
             covestimator = SimpleCovariance()) where T<:Real
    μp = vec(mean(Xp, dims=2))
    μn = vec(mean(Xn, dims=2))
    Zp = Xp .- μp
    Zn = Xn .- μn
    Cp = calcscattermat(Zp, covestimator)
    Cn = calcscattermat(Zn, covestimator)
    ldacov(Cp, Cn, μp, μn)
end


#### Multiclass LDA Stats

mutable struct MulticlassLDAStats{T<:Real}
    dim::Int              # sample dimensions
    nclasses::Int         # number of classes
    cweights::Vector{T}   # class weights
    tweight::T            # total sample weight
    mean::Vector{T}       # overall sample mean
    cmeans::Matrix{T}     # class-specific means
    Sw::Matrix{T}         # within-class scatter matrix
    Sb::Matrix{T}         # between-class scatter matrix
end

mean(S::MulticlassLDAStats) = S.mean
classweights(S::MulticlassLDAStats) = S.cweights
classmeans(S::MulticlassLDAStats) = S.cmeans

withclass_scatter(S::MulticlassLDAStats) = S.Sw
betweenclass_scatter(S::MulticlassLDAStats) = S.Sb

function MulticlassLDAStats(cweights::Vector{T},
                            mean::Vector{T},
                            cmeans::Matrix{T},
                            Sw::Matrix{T},
                            Sb::Matrix{T}) where T<:Real
    d, nc = size(cmeans)
    length(mean) == d || throw(DimensionMismatch("Incorrect length of mean"))
    length(cweights) == nc || throw(DimensionMismatch("Incorrect length of cweights"))
    tw = sum(cweights)
    size(Sw) == (d, d) || throw(DimensionMismatch("Incorrect size of Sw"))
    size(Sb) == (d, d) || throw(DimensionMismatch("Incorrect size of Sb"))
    MulticlassLDAStats(d, nc, cweights, tw, mean, cmeans, Sw, Sb)
end

function multiclass_lda_stats(nc::Int, X::DenseMatrix{T}, y::AbstractVector{Int};
                              covestimator=SimpleCovariance()) where T<:Real
    # check sizes
    d = size(X, 1)
    n = size(X, 2)
    n ≥ nc || throw(ArgumentError("The number of samples is less than the number of classes"))
    length(y) == n || throw(DimensionMismatch("Inconsistent array sizes."))

    # compute class-specific weights and means
    cmeans, cweights, Z = center(X, y, nc)

    Sw = calcscattermat(Z, covestimator)

    # compute between-class scattering
    mean = cmeans * (cweights ./ T(n))
    U = rmul!(cmeans .- mean, Diagonal(sqrt.(cweights)))
    Sb = calcscattermat(U, covestimator)

    return MulticlassLDAStats(Vector{T}(cweights), mean, cmeans, Sw, Sb)
end


#### Multiclass LDA

mutable struct MulticlassLDA{T<:Real}
    proj::Matrix{T}
    pmeans::Matrix{T}
    stats::MulticlassLDAStats{T}
end

indim(M::MulticlassLDA) = size(M.proj, 1)
outdim(M::MulticlassLDA) = size(M.proj, 2)

projection(M::MulticlassLDA) = M.proj

mean(M::MulticlassLDA) = mean(M.stats)
classmeans(M::MulticlassLDA) = classmeans(M.stats)
classweights(M::MulticlassLDA) = classweights(M.stats)

withclass_scatter(M::MulticlassLDA) = withclass_scatter(M.stats)
betweenclass_scatter(M::MulticlassLDA) = betweenclass_scatter(M.stats)

transform(M::MulticlassLDA, x::AbstractVecOrMat{T}) where T<:Real = M.proj'x

function fit(::Type{MulticlassLDA}, nc::Int, X::DenseMatrix{T}, y::AbstractVector{Int};
             method::Symbol=:gevd,
             outdim::Int=min(size(X,1), nc-1),
             regcoef::T=T(1.0e-6),
             covestimator=SimpleCovariance()) where T<:Real

    multiclass_lda(multiclass_lda_stats(nc, X, y; covestimator=covestimator);
                   method=method,
                   regcoef=regcoef,
                   outdim=outdim)
end

function multiclass_lda(S::MulticlassLDAStats{T};
                        method::Symbol=:gevd,
                        outdim::Int=min(size(X,1), S.nclasses-1),
                        regcoef::T=T(1.0e-6)) where T<:Real

    P = mclda_solve(S.Sb, S.Sw, method, outdim, regcoef)
    MulticlassLDA(P, P'S.cmeans, S)
end

mclda_solve(Sb::DenseMatrix{T}, Sw::DenseMatrix{T}, method::Symbol, p::Int, regcoef::T) where T<:Real =
    mclda_solve!(copy(Sb), copy(Sw), method, p, regcoef)

function mclda_solve!(Sb::Matrix{T},
                      Sw::Matrix{T},
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

function _lda_whitening!(C::Matrix{T}, regcoef::T) where T<:Real
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

# When the dimensionality is much higher than the number of samples,
# it makes more sense to perform LDA on the space spanned by the
# within-group scatter.

struct SubspaceLDA{T<:Real}
    projw::Matrix{T}
    projLDA::Matrix{T}
    λ::Vector{T}
    cmeans::Matrix{T}
    cweights::Vector{Int}
end

indim(M::SubspaceLDA) = size(M.projw,1)
outdim(M::SubspaceLDA) = size(M.projLDA, 2)

projection(M::SubspaceLDA) = M.projw * M.projLDA

mean(M::SubspaceLDA) = vec(sum(M.cmeans * Diagonal(M.cweights / sum(M.cweights)), dims=2))
classmeans(M::SubspaceLDA) = M.cmeans
classweights(M::SubspaceLDA) = M.cweights

transform(M::SubspaceLDA, x) = M.projLDA' * (M.projw' * x)

fit(::Type{F}, X::AbstractMatrix{T}, nc::Int, label::AbstractVector{Int}) where {T<:Real, F<:SubspaceLDA} =
    fit(F, X, label, nc)

function fit(::Type{F}, X::AbstractMatrix{T},
             label::AbstractVector{Int},
             nc=maximum(label);
             normalize::Bool=false) where {T<:Real, F<:SubspaceLDA}
    d, n = size(X, 1), size(X, 2)
    n ≥ nc || throw(ArgumentError("The number of samples is less than the number of classes"))
    length(label) == n || throw(DimensionMismatch("Inconsistent array sizes."))
    # Compute centroids, class weights, and deviation from centroids
    # Note Sb = Hb*Hb', Sw = Hw*Hw'
    cmeans, cweights, Hw = center(X, label, nc)
    dmeans = cmeans .- (normalize ? mean(cmeans, dims=2) : cmeans * (cweights / n))
    Hb = normalize ? dmeans : dmeans * Diagonal(sqrt.(cweights))
    if normalize
        Hw /= sqrt(n)
    end
    # Project to the subspace spanned by the within-class scatter
    # (essentially, PCA before LDA)
    Uw, Σw, _ = svd(Hw, full=false)
    keep = Σw .> sqrt(eps(T)) * maximum(Σw)
    projw = Uw[:,keep]
    pHb = projw' * Hb
    pHw = projw' * Hw
    λ, G = lda_gsvd(pHb, pHw, cweights)
    SubspaceLDA(projw, G, λ, cmeans, cweights)
end

# Reference: Howland & Park (2006), "Generalizing discriminant analysis
# using the generalized singular value decomposition", IEEE
# Trans. Patt. Anal. & Mach. Int., 26: 995-1006.
function lda_gsvd(Hb::AbstractMatrix{T}, Hw::AbstractMatrix{T}, cweights::AbstractVector{Int}) where T<:Real
    nc = length(cweights)
    K = vcat(Hb', Hw')
    P, R, Q = svd(K, full=false)
    keep = R .> sqrt(eps(T))*maximum(R)
    R = R[keep]
    Pk = P[1:nc, keep]
    U, ΣA, W = svd(Pk)
    ncnz = sum(cweights .> 0)
    G = Q[:,keep]*(Diagonal(1 ./ R) * W[:,1:ncnz-1])
    # Normalize
    Gw = G' * Hw
    nrm = Gw * Gw'
    G = G ./ reshape(sqrt.(diag(nrm)), 1, ncnz-1)
    # Also get the eigenvalues
    Gw = G' * Hw
    Gb = G' * Hb
    λ = diag(Gb * Gb')./diag(Gw * Gw')
    λ, G
end

function center(X::AbstractMatrix{T}, label::AbstractVector{Int}, nc=maximum(label)) where T<:Real
    d, n = size(X,1), size(X,2)
    # Calculate the class weights and means
    cmeans = zeros(T, d, nc)
    cweights = zeros(Int, nc)
    for j = 1:n
        k = label[j]
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
        k = label[j]
        for i = 1:d
            dX[i,j] = X[i,j] - cmeans[i,k]
        end
    end
    cmeans, cweights, dX
end
