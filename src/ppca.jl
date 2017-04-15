# Probabilistic Principal Component Analysis

"""Probabilistic PCA type"""
immutable PPCA{T<:AbstractFloat}
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    W::Matrix{T}          # weight matrix: of size d x p
    σ²::T                 # residual variance
end

## properties

indim(M::PPCA) = size(M.W, 1)
outdim(M::PPCA) = size(M.W, 2)

Base.mean(M::PPCA) = fullmean(indim(M), M.mean)
projection(M::PPCA) = svdfact(M.W)[:U] # recover principle components from the weight matrix
Base.var(M::PPCA) = M.σ²
loadings(M::PPCA) = M.W

## use

function transform{T<:AbstractFloat}(m::PPCA{T}, x::AbstractVecOrMat{T})
    xn = centralize(x, m.mean)
    W  = m.W
    M = W'W .+ m.σ²*eye(size(m.W,2))
    return inv(M)*m.W'*xn
end

function reconstruct{T<:AbstractFloat}(m::PPCA{T}, z::AbstractVecOrMat{T})
    W  = m.W
    WTW = W'W
    M  = WTW .+ var(m)*eye(size(WTW,1))
    return W*inv(WTW)*M*z .+ mean(m)
end

## show

function Base.show(io::IO, M::PPCA)
    print(io, "Probabilistic PCA(indim = $(indim(M)), outdim = $(outdim(M)), σ² = $(var(M)))")
end

## core algorithms

function ppcaml{T<:AbstractFloat}(Z::DenseMatrix{T}, mean::Vector{T};
                maxoutdim::Int=size(Z,1)-1,
                tol::Real=1.0e-6) # convergence tolerance

    check_pcaparams(size(Z,1), mean, maxoutdim, 1.)

    d = size(Z,1)

    # SVD decomposition
    Svd = svdfact(Z)
    λ = Svd[:S]::Vector{T}
    ord = sortperm(λ; rev=true)
    V = λ[ord]

    # filter 0 eigenvalues and adjust number of latent dimensions
    idxs = find(V .< tol)
    l = length(idxs)
    l = l == 0 ? maxoutdim : l

    # variance "loss" in the projection
    σ² = sum(V[l+1:end])/(d-l)

    @inbounds for i in 1:l
        V[i] = sqrt(V[i] - σ²)
    end

    U = Svd[:U]::Matrix{T}
    W = U[:,ord[1:l]]*diagm(V[1:l])

    return PPCA(mean, W, σ²)
end

function ppcaem{T<:AbstractFloat}(S::DenseMatrix{T}, mean::Vector{T}, n::Int;
                maxoutdim::Int=size(S,1)-1,
                tol::Real=1.0e-6,   # convergence tolerance
                tot::Integer=1000)  # maximum number of iterations

    check_pcaparams(size(S,1), mean, maxoutdim, 1.)

    d = size(S,1)
    q = maxoutdim
    W = eye(d,q)
    σ² = 0.
    M⁻¹ = inv(W'W .+ σ²*eye(q))

    i = 1
    L_old = 0.
    while i < tot
        # EM-steps
        SW = S*W
        W⁺  = SW*inv(σ²*eye(q) + M⁻¹*W'*SW)
        σ²⁺ = trace(S - SW*M⁻¹*(W⁺)')/d
        # new parameters
        W = W⁺
        σ² = σ²⁺
        # log likelihood
        C = W*W'.+ σ²*eye(d)
        M⁻¹ = inv(W'*W .+ σ²*eye(q))
        C⁻¹ = (eye(d) - W*M⁻¹*W')/σ²
        L = (-n/2)*(log(det(C)) + trace(C⁻¹*S))  # (-n/2)*d*log(2π) omitted
        # println("$i] ΔL: $(abs(L_old - L)), L: $L")
        if abs(L_old - L) < tol
            break
        end
        L_old = L
        i += 1
    end

    return PPCA(mean, W, σ²)
end

function bayespca{T<:AbstractFloat}(S::DenseMatrix{T}, mean::Vector{T}, n::Int;
                 maxoutdim::Int=size(S,1)-1,
                 tol::Real=1.0e-6,   # convergence tolerance
                 tot::Integer=1000)  # maximum number of iterations

    check_pcaparams(size(S,1), mean, maxoutdim, 1.)

    d = size(S,1)
    q = maxoutdim
    W = eye(d,q) #rand(d,q)
    wnorm = zeros(q)
    σ² = 0.
    M = W'*W .+ σ²*eye(q)
    M⁻¹ = inv(M)
    α = zeros(q)

    i = 1
    L_old = 0.
    while i < tot
        # EM-steps
        SW = S*W
        W⁺  = SW*inv(σ²*(eye(q)+diagm(α)*M/n) + M⁻¹*W'*SW)
        σ²⁺ = trace(S - SW*M⁻¹*(W⁺)')/d
        # new parameters
        W = W⁺
        σ² = σ²⁺
        @inbounds for j in 1:q
            wnorm[j] = norm(W[:,j])^2
            α[j] = wnorm[j] < eps() ? maxintfloat(Float64) : d/wnorm[j]
        end
        # log likelihood
        C = W*W'.+ σ²*eye(d)
        M = W'*W .+ σ²*eye(q)
        M⁻¹ = inv(M)
        C⁻¹ = (eye(d) - W*M⁻¹*W')/σ²
        L = (-n/2)*(log(det(C)) + trace(C⁻¹*S))  # (-n/2)*d*log(2π) omitted
        # println("$i] ΔL: $(abs(L_old - L)), L: $L")
        if abs(L_old - L) < tol
            break
        end
        L_old = L
        i += 1
    end

    return PPCA(mean, W[:,wnorm .> 0.], σ²)
end

## interface functions

function fit{T<:AbstractFloat}(::Type{PPCA}, X::DenseMatrix{T};
             method::Symbol=:ml,
             maxoutdim::Int=size(X,1)-1,
             mean=nothing,
             tol::Real=1.0e-6,   # convergence tolerance
             tot::Integer=1000)  # maximum number of iterations

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
        S = Base.covm(X, isempty(mv) ? 0 : mv, 2)
        if method == :em
            M = ppcaem(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot)
        elseif method == :bayes
            M = bayespca(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot)
        end
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::PPCA
end
