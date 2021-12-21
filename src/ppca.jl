# Probabilistic Principal Component Analysis

"""Probabilistic PCA type"""
struct PPCA{T<:Real}
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    W::Matrix{T}          # weight matrix: of size d x p
    σ²::T                 # residual variance
end

## properties

indim(M::PPCA) = size(M.W, 1)
outdim(M::PPCA) = size(M.W, 2)

mean(M::PPCA) = fullmean(indim(M), M.mean)
projection(M::PPCA) = svd(M.W).U # recover principle components from the weight matrix
var(M::PPCA) = M.σ²
loadings(M::PPCA) = M.W

## use

function transform(m::PPCA, x::AbstractVecOrMat{<:Real})
    xn = centralize(x, m.mean)
    W  = m.W
    n = outdim(m)
    M = W'W + m.σ² * I
    return inv(M)*m.W'*xn
end

function reconstruct(m::PPCA, z::AbstractVecOrMat{<:Real})
    W  = m.W
    WTW = W'W
    n = outdim(m)
    M  = WTW + var(m) * I
    return W*inv(WTW)*M*z .+ mean(m)
end

## show

function Base.show(io::IO, M::PPCA)
    print(io, "Probabilistic PCA(indim = $(indim(M)), outdim = $(outdim(M)), σ² = $(var(M)))")
end

## core algorithms

function ppcaml(Z::AbstractMatrix{T}, mean::Vector{T};
                tol::Real=1.0e-6, # convergence tolerance
                maxoutdim::Int=size(Z,1)-1) where {T<:Real}

    check_pcaparams(size(Z,1), mean, maxoutdim, 1.)

    d = size(Z,1)

    # SVD decomposition
    Svd = svd(Z)
    λ = Svd.S
    ord = sortperm(λ; rev=true)
    V = λ[ord]

    # filter 0 eigenvalues and adjust number of latent dimensions
    idxs = findall(V .< tol)
    l = length(idxs)
    l = l == 0 ? maxoutdim : l

    # variance "loss" in the projection
    σ² = sum(V[l+1:end])/(d-l)

    @inbounds for i in 1:l
        V[i] = sqrt(V[i] - σ²)
    end

    W = Svd.U[:,ord[1:l]]*diagm(0 => V[1:l])

    return PPCA(mean, W, σ²)
end

function ppcaem(S::AbstractMatrix{T}, mean::Vector{T}, n::Int;
                maxoutdim::Int=size(S,1)-1,
                tol::Real=1.0e-6,   # convergence tolerance
                maxiter::Integer=1000) where {T<:Real}

    check_pcaparams(size(S,1), mean, maxoutdim, 1.)

    d = size(S,1)
    q = maxoutdim
    Iq = Matrix{T}(I, q, q)
    Id = Matrix{T}(I, d, d)
    W = Matrix{T}(I, d, q)
    σ² = zero(T)
    M⁻¹ = inv(W'W .+ σ² * Iq)

    i = 1
    L_old = 0.
    chg = NaN
    converged = false
    while i < maxiter
        # EM-steps
        SW = S*W
        W⁺  = SW*inv(σ²*Iq + M⁻¹*W'*SW)
        σ²⁺ = tr(S - SW*M⁻¹*(W⁺)')/d
        # new parameters
        W = W⁺
        σ² = σ²⁺
        # log likelihood
        C = W*W'.+ σ²*Id
        M⁻¹ = inv(W'*W .+ σ²*Iq)
        C⁻¹ = (Id - W*M⁻¹*W')/σ²
        L = (-n/2)*(log(det(C)) + tr(C⁻¹*S))  # (-n/2)*d*log(2π) omitted

        @debug "Likelihood" iter=i L=L ΔL=abs(L_old - L)
        chg = abs(L_old - L)
        if chg < tol
            converged = true
            break
        end
        L_old = L
        i += 1
    end
    converged || throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))

    return PPCA(mean, W, σ²)
end

function bayespca(S::AbstractMatrix{T}, mean::Vector{T}, n::Int;
                 maxoutdim::Int=size(S,1)-1,
                 tol::Real=1.0e-6,   # convergence tolerance
                 maxiter::Integer=1000) where {T<:Real}

    check_pcaparams(size(S,1), mean, maxoutdim, 1.)

    d = size(S,1)
    q = maxoutdim
    Iq = Matrix{T}(I, q, q)
    Id = Matrix{T}(I, d, d)
    W = Matrix{T}(I, d, q)
    wnorm = zeros(T, q)
    σ² = zero(T)
    M = W'*W .+ σ²*Iq
    M⁻¹ = inv(M)
    α = zeros(T, q)

    i = 1
    chg = NaN
    L_old = 0.
    converged = false
    while i < maxiter
        # EM-steps
        SW = S*W
        W⁺  = SW*inv(σ²*(Iq+diagm(0=>α)*M/n) + M⁻¹*W'*SW)
        σ²⁺ = tr(S - SW*M⁻¹*(W⁺)')/d
        # new parameters
        W = W⁺
        σ² = σ²⁺
        @inbounds for j in 1:q
            wnorm[j] = norm(W[:,j])^2
            α[j] = wnorm[j] < eps() ? maxintfloat(Float64) : d/wnorm[j]
        end
        # log likelihood
        C = W*W'.+ σ²*Id
        M = W'*W .+ σ²*Iq
        M⁻¹ = inv(M)
        C⁻¹ = (Id - W*M⁻¹*W')/σ²
        L = (-n/2)*(log(det(C)) + tr(C⁻¹*S))  # (-n/2)*d*log(2π) omitted

        @debug "Likelihood" iter=i L=L ΔL=abs(L_old - L)
        chg = abs(L_old - L)
        if chg < tol
            converged = true
            break
        end
        L_old = L
        i += 1
    end
    converged || throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))

    return PPCA(mean, W[:,wnorm .> 0.], σ²)
end

## interface functions

function fit(::Type{PPCA}, X::AbstractMatrix{T};
             method::Symbol=:ml,
             maxoutdim::Int=size(X,1)-1,
             mean=nothing,
             tol::Real=1.0e-6,   # convergence tolerance
             maxiter::Integer=1000) where {T<:Real}

    @assert !SparseArrays.issparse(X) "Use Kernel PCA for sparse arrays"

    d, n = size(X)

    # process mean
    mv = preprocess_mean(X, mean)
    if !(isempty(mv) || length(mv) == d)
        throw(DimensionMismatch("Dimensions of weight matrix and mean are inconsistent."))
    end

    if method == :ml
        Z = centralize(X, mv)
        M = ppcaml(Z, mv, maxoutdim=maxoutdim, tol=tol)
    elseif method == :em || method == :bayes
        S = covm(X, isempty(mv) ? 0 : mv, 2)
        if method == :em
            M = ppcaem(S, mv, n, maxoutdim=maxoutdim, tol=tol, maxiter=maxiter)
        elseif method == :bayes
            M = bayespca(S, mv, n, maxoutdim=maxoutdim, tol=tol, maxiter=maxiter)
        end
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::PPCA
end
