# Linear Discriminant Analysis

#### Type to represent a linear discriminant functional

abstract Discriminant{T}

immutable LinearDiscriminant{T<:AbstractFloat} <: Discriminant{T}
    w::Vector{T}
    b::T
end

length(f::LinearDiscriminant) = length(f.w)

evaluate(f::LinearDiscriminant, x::AbstractVector) = dot(f.w, x) + f.b

function evaluate(f::LinearDiscriminant, X::AbstractMatrix)
    R = At_mul_B(X, f.w)
    if f.b != 0
        broadcast!(+, R, R, f.b)
    end
    return R
end

predict(f::Discriminant, x::AbstractVector) = evaluate(f, x) > 0

predict(f::Discriminant, X::AbstractMatrix) = (Y = evaluate(f, X); Bool[y > 0 for y in Y])


#### function to solve linear discriminant

function ldacov{T<:AbstractFloat}(C::DenseMatrix{T},
                μp::DenseVector{T},
                μn::DenseVector{T})

    w = cholfact(C) \ (μp - μn)
    ap = w ⋅ μp
    an = w ⋅ μn
    c = 2 / (ap - an)
    LinearDiscriminant(scale!(w, c), 1 - c * ap)
end

ldacov{T<:AbstractFloat}(Cp::DenseMatrix{T},
       Cn::DenseMatrix{T},
       μp::DenseVector{T},
       μn::DenseVector{T}) = ldacov(Cp + Cn, μp, μn)

#### interface functions

function fit{T<:AbstractFloat}(::Type{LinearDiscriminant}, Xp::DenseMatrix{T}, Xn::DenseMatrix{T})
    μp = vec(mean(Xp, 2))
    μn = vec(mean(Xn, 2))
    Zp = Xp .- μp
    Zn = Xn .- μn
    Cp = A_mul_Bt(Zp, Zp)
    Cn = A_mul_Bt(Zn, Zn)
    ldacov(Cp, Cn, μp, μn)
end


#### Multiclass LDA Stats

type MulticlassLDAStats{T<:AbstractFloat}
    dim::Int              # sample dimensions
    nclasses::Int         # number of classes
    cweights::Vector{T}   # class weights
    tweight::T            # total sample weight
    mean::Vector{T}       # overall sample mean
    cmeans::Matrix{T}     # class-specific means
    Sw::Matrix{T}         # within-class scatter matrix
    Sb::Matrix{T}         # between-class scatter matrix
end

Base.mean(S::MulticlassLDAStats) = S.mean
classweights(S::MulticlassLDAStats) = S.cweights
classmeans(S::MulticlassLDAStats) = S.cmeans

withclass_scatter(S::MulticlassLDAStats) = S.Sw
betweenclass_scatter(S::MulticlassLDAStats) = S.Sb

function MulticlassLDAStats{T<:AbstractFloat}(cweights::Vector{T},
                            mean::Vector{T},
                            cmeans::Matrix{T},
                            Sw::Matrix{T},
                            Sb::Matrix{T})
    d, nc = size(cmeans)
    length(mean) == d || throw(DimensionMismatch("Incorrect length of mean"))
    length(cweights) == nc || throw(DimensionMismatch("Incorrect length of cweights"))
    tw = sum(cweights)
    size(Sw) == (d, d) || throw(DimensionMismatch("Incorrect size of Sw"))
    size(Sb) == (d, d) || throw(DimensionMismatch("Incorrect size of Sb"))
    MulticlassLDAStats(d, nc, cweights, tw, mean, cmeans, Sw, Sb)
end

function multiclass_lda_stats{T<:AbstractFloat}(nc::Int, X::DenseMatrix{T}, y::AbstractVector{Int})
    # check sizes
    d = size(X, 1)
    n = size(X, 2)
    n ≥ nc || throw(ArgumentError("The number of samples is less than the number of classes"))
    length(y) == n || throw(DimensionMismatch("Inconsistent array sizes."))

    # compute class-specific weights and means
    cweights = zeros(nc)
    cmeans = zeros(d, nc)
    for j = 1:n
        @inbounds c = y[j]
        1 <= c <= nc || error("class label must be in [1, nc].")
        cweights[c] += 1
        v = view(cmeans,:,c)
        x = view(X,:,j)
        for i = 1:d
            @inbounds v[i] += x[i]
        end
    end
    for j = 1:nc
        @inbounds cw = cweights[j]
        cw > 0 || error("The $(j)-th class has no sample.")
        scale!(view(cmeans,:,j), inv(cw))
    end

    # compute within-class scattering
    Z = Array(T, d, n)
    for j = 1:n
        z = view(Z,:,j)
        x = view(X,:,j)
        v = view(cmeans,:,y[j])
        for i = 1:d
            z[i] = x[i] - v[i]
        end
    end
    Sw = A_mul_Bt(Z, Z)

    # compute between-class scattering
    mean = cmeans * (cweights ./ n)
    U = scale!(cmeans .- mean, sqrt(cweights))
    Sb = A_mul_Bt(U, U)

    return MulticlassLDAStats(cweights, mean, cmeans, Sw, Sb)
end


#### Multiclass LDA

type MulticlassLDA{T<:AbstractFloat}
    proj::Matrix{T}
    pmeans::Matrix{T}
    stats::MulticlassLDAStats{T}
end

indim(M::MulticlassLDA) = size(M.proj, 1)
outdim(M::MulticlassLDA) = size(M.proj, 2)

projection(M::MulticlassLDA) = M.proj

Base.mean(M::MulticlassLDA) = mean(M.stats)
classmeans(M::MulticlassLDA) = classmeans(M.stats)
classweights(M::MulticlassLDA) = classweights(M.stats)

withclass_scatter(M::MulticlassLDA) = withclass_scatter(M.stats)
betweenclass_scatter(M::MulticlassLDA) = betweenclass_scatter(M.stats)

transform{T<:AbstractFloat}(M::MulticlassLDA, x::AbstractVecOrMat{T}) = M.proj'x

function fit{T<:AbstractFloat}(::Type{MulticlassLDA}, nc::Int, X::DenseMatrix{T}, y::AbstractVector{Int};
             method::Symbol=:gevd,
             outdim::Int=min(size(X,1), nc-1),
             regcoef::T=1.0e-6)

    multiclass_lda(multiclass_lda_stats(nc, X, y);
                   method=method,
                   regcoef=regcoef,
                   outdim=outdim)
end

function multiclass_lda{T<:AbstractFloat}(S::MulticlassLDAStats{T};
                        method::Symbol=:gevd,
                        outdim::Int=min(size(X,1), S.nclasses-1),
                        regcoef::T=1.0e-6)

    P = mclda_solve(S.Sb, S.Sw, method, outdim, regcoef)
    MulticlassLDA(P, P'S.cmeans, S)
end

mclda_solve{T<:AbstractFloat}(Sb::DenseMatrix{T}, Sw::DenseMatrix{T}, method::Symbol, p::Int, regcoef::T) =
    mclda_solve!(copy(Sb), copy(Sw), method, p, regcoef)

function mclda_solve!{T<:AbstractFloat}(Sb::Matrix{T},
                      Sw::Matrix{T},
                      method::Symbol, p::Int, regcoef::T)

    p <= size(Sb, 1) || throw(ArgumentError("p cannot exceed sample dimension."))

    if method == :gevd
        regularize_symmat!(Sw, regcoef)
        E = eigfact!(Symmetric(Sb), Symmetric(Sw))
        ord = sortperm(E.values; rev=true)
        P = E.vectors[:, ord[1:p]]

    elseif method == :whiten
        W = _lda_whitening!(Sw, regcoef)
        wSb = At_mul_B(W, Sb * W)
        Eb = eigfact!(Symmetric(wSb))
        ord = sortperm(Eb.values; rev=true)
        P = W * Eb.vectors[:, ord[1:p]]

    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    return P::Matrix{T}
end

function _lda_whitening!{T<:AbstractFloat}(C::Matrix{T}, regcoef::T)
    n = size(C,1)
    E = eigfact!(Symmetric(C))
    v = E.values
    a = regcoef * maximum(v)
    for i = 1:n
        @inbounds v[i] = 1.0 / sqrt(v[i] + a)
    end
    return scale!(E.vectors, v)
end

