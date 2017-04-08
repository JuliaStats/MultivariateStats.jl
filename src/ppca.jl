# Probabilistic Principal Component Analysis

"""Probabilistic PCA type"""
type ProbPCA{T<:AbstractFloat}
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    W::Matrix{T}          # weight matrix: of size d x p
    σ²::T                 # residual variance

    function (::Type{ProbPCA}){T}(mean::Vector{T}, W::Matrix{T}, var::T)
        d = size(W, 1)
        (isempty(mean) || length(mean) == d) ||
            throw(DimensionMismatch("Dimensions of weight matrix and mean are inconsistent."))
        new{T}(mean, W, var)
    end
end

## properties

MultivariateStats.indim(M::ProbPCA) = size(M.W, 1)
MultivariateStats.outdim(M::ProbPCA) = size(M.W, 2)

Base.mean(M::ProbPCA) = MultivariateStats.fullmean(indim(M), M.mean)
projection(M::ProbPCA) = M.W
Base.var(M::ProbPCA) = M.σ²

## use

function MultivariateStats.transform{T<:AbstractFloat}(m::ProbPCA{T}, x::AbstractVecOrMat{T})
    xn = centralize(x, m.mean)
    W  = m.W
    M = W'W .+ m.σ²*eye(size(m.W,2))
    return inv(M)*m.W'*xn
end

function MultivariateStats.reconstruct{T<:AbstractFloat}(m::ProbPCA{T}, z::AbstractVecOrMat{T})
    W  = m.W
    WTW = W'W
    M  = WTW .+ var(m)*eye(size(WTW,1))
    return W*inv(WTW)*M*z .+ mean(m)
end

## show

function Base.show(io::IO, M::ProbPCA)
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

    return ProbPCA(mean, W, σ²)
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

    return ProbPCA(mean, W, σ²)
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

    return ProbPCA(mean, W[:,wnorm .> 0.], σ²)
end

## interface functions

function MultivariateStats.fit{T<:AbstractFloat}(::Type{ProbPCA}, X::DenseMatrix{T};
             method::Symbol=:ml,
             maxoutdim::Int=size(X,1)-1,
             mean=nothing,
             tol::Real=1.0e-6,   # convergence tolerance
             tot::Integer=1000)  # maximum number of iterations

    d, n = size(X)

    # process mean
    mv = MultivariateStats.preprocess_mean(X, mean)

    if method == :ml
        Z = centralize(X, mv)
        M = ppcaml(Z, mv, maxoutdim=maxoutdim, tol=tol)
    elseif method == :em
        S = Base.covm(X, isempty(mv) ? 0 : mv, 2)
        M = ppcaem(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot)
    elseif method == :bayes
        S = Base.covm(X, isempty(mv) ? 0 : mv, 2)
        M = bayespca(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::ProbPCA
end