# Factor Analysis

"""
This type contains factor analysis model parameters.
"""
struct FactorAnalysis{T<:Real} <: LatentVariableDimensionalityReduction
    mean::Vector{T}       # sample mean: of length d
    W::Matrix{T}          # factor loadings matrix: of size d x p
    Ψ::Vector{T}          # noise covariance: diagonal of size d x d
end

"""
    size(M::PPCA)

Returns a tuple with values of the input dimension ``d``, *i.e* the dimension of
the observation space, and the output dimension ``p``, *i.e* the dimension of
the principal subspace.
"""
size(M::FactorAnalysis) = size(M.W)

"""
    mean(M::PPCA)

Get the mean vector (of length ``d``).
"""
mean(M::FactorAnalysis) = fullmean(size(M)[1], M.mean)

"""
    projection(M::FactorAnalysis)

Recovers principle components from the weight matrix of the model `M`.
"""
projection(M::FactorAnalysis) = svd(M.W).U

"""
    cov(M::FactorAnalysis)

Returns the covariance of the model `M`.
"""
cov(M::FactorAnalysis) = M.W*M.W' + Diagonal(M.Ψ)

"""
    var(M::FactorAnalysis)

Returns the variance of the model `M`.
"""
var(M::FactorAnalysis) = M.Ψ

"""
    loadings(M::FactorAnalysis)

Returns the factor loadings matrix of the model `M`.
"""
loadings(M::FactorAnalysis) = M.W

## use

"""
    predict(M::FactorAnalysis, x)

Transform observations `x` into latent variables. Here, `x` can be either a vector
of length `d` or a matrix where each column is an observation.
"""
function predict(m::FactorAnalysis, x::AbstractVecOrMat{<:Real})
    xn = centralize(x, mean(m))
    W = m.W
    WᵀΨ⁻¹ = W'*diagm(0 => 1 ./ m.Ψ)  # (q x d) * (d x d) = (q x d)
    return inv(I+WᵀΨ⁻¹*W)*(WᵀΨ⁻¹*xn)  # (q x q) * (q x d) * (d x 1) = (q x 1)
end

"""
    reconstruct(M::FactorAnalysis, z)

Approximately reconstruct observations from the latent variable given in `z`.
Here, `z` can be either a vector of length ``p`` or a matrix where each column gives
the latent variables for an observation.
"""
function reconstruct(m::FactorAnalysis, z::AbstractVecOrMat{<:Real})
    W  = m.W
    # ΣW(W'W)⁻¹z+μ = ΣW(W'W)⁻¹W'Σ⁻¹(x-μ)+μ = Σ(WW⁻¹)((W')⁻¹W')Σ⁻¹(x-μ)+μ
    # = ΣΣ⁻¹(x-μ)+μ = (x-μ)+μ = x
    return cov(m)*W*inv(W'W)*z .+ mean(m)
end

## show

function show(io::IO, M::FactorAnalysis)
    print(io, "Factor Analysis(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## core algorithms

"""
    faem(S, mean, n; ...)

Performs factor analysis using an expectation-maximization algorithm for a given sample covariance matrix `S`[^2].

**Parameters**
- `S`: The sample covariance matrix.
- `mean`: The mean vector of original samples, which can be a vector of length ``d``,
or an empty vector indicating a zero mean.
- `n`: The number of observations.

Returns the resultant [`FactorAnalysis`](@ref) model.

**Note:** This function accepts two keyword arguments: `maxoutdim`,`tol`, and `maxiter`.
"""
function faem(S::AbstractMatrix{T}, mv::Vector{T}, n::Int;
             maxoutdim::Int=size(X,1)-1,
             tol::Real=1.0e-6,   # convergence tolerance
             maxiter::Integer=1000) where T<:Real

    d = size(S,1)
    q = maxoutdim
    W = Matrix{T}(I,d,q)
    Ψ = fill(T(0.01),d)

    L_old = 0.
    for c in 1:maxiter
        # EM-steps
        Ψ⁻¹W = diagm(0 => 1 ./ Ψ)*W
        SΨ⁻¹W = S*Ψ⁻¹W
        H = SΨ⁻¹W*inv(I + W'*Ψ⁻¹W)
        W⁺ = SΨ⁻¹W*inv(I + H'*Ψ⁻¹W)
        Ψ⁺ = diag(S - H*(W⁺)')
        # new parameters
        W = W⁺
        Ψ = Ψ⁺
        # log likelihood
        Ψ⁻¹ = diagm(0 => 1 ./ Ψ)
        WᵀΨ⁻¹ = W'*Ψ⁻¹
        logdetΣ = sum(log, Ψ) + logabsdet(I + WᵀΨ⁻¹*W)[1]
        Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*inv(I + WᵀΨ⁻¹*W)*WᵀΨ⁻¹
        L = (-n/2)*(d*log(2π) + logdetΣ + tr(Σ⁻¹*S))
        @debug "Likelihood" iter=c L=L ΔL=abs(L_old - L)
        if abs(L_old - L) < tol
            break
        end
        L_old = L
    end

    return FactorAnalysis(mv, W, Ψ)
end

"""
    facm(S, mean, n; ...)

Performs factor analysis using a fast conditional maximization algorithm for a given sample covariance matrix `S`[^3].

**Parameters**
- `S`: The sample covariance matrix.
- `mean`: The mean vector of original samples, which can be a vector of length ``d``,
or an empty vector indicating a zero mean.
- `n`: The number of observations.

Returns the resultant [`FactorAnalysis`](@ref) model.

**Note:** This function accepts two keyword arguments: `maxoutdim`,`tol`, `maxiter`, and `η`.
"""
function facm(S::AbstractMatrix{T}, mv::Vector{T}, n::Int;
             maxoutdim::Int=size(X,1)-1,
             tol::Real=1.0e-6,   # convergence tolerance
             η = tol,            # variance low bound
             maxiter::Integer=1000) where T<:Real

    d = size(S,1)

    q = maxoutdim
    W = Matrix{T}(I,d,q)
    Ψ = fill(T(0.01),d)
    V = zeros(T,q)
    eᵢeᵢ = zeros(T,d,d)
    Bᵢ⁻¹ = zeros(T,d,d)
    addconst = d*log(2π)

    L_old = 0.
    for c in 1:maxiter
        # CM-step 1
        Ψ⁻ʰ = diagm(0 => 1 ./ sqrt.(Ψ))
        S⁺ = Ψ⁻ʰ*S*Ψ⁻ʰ

        F = eigen(S⁺)
        λ = real(F.values)
        ord = sortperm(λ, rev=true)
        λ = λ[ord]

        q′ = λ[q] > 0 ? q : findlast(λ .> 0)

        λq = 0.0
        @inbounds for i in 1:q′
            λq += log(λ[i]) - λ[i] + 1.
            V[i] = sqrt(λ[i] - 1.)
        end

        L = (addconst + log(prod(Ψ)) + tr(S⁺) + λq)*(-n/2)

        U = convert(Matrix{T}, F.vectors[:,ord[1:q′]])
        W = U*diagm(0 => V[1:q′]) # set new parameter

        # CM-step 2
        @inbounds for i in 1:q′
            V[i] = 1 ./ λ[i] - 1.
        end
        eBe⁻¹ = ωᵢᵗ⁺¹ = 0.0
        for i in 1:d
            if i == 1
                Bᵢ⁻¹ = U*diagm(0 => V[1:q′])*U' + I
            else
                eᵢeᵢ[i-1,i-1] = 1.
                Bᵢ⁻¹ = Bᵢ⁻¹ -  ωᵢᵗ⁺¹*Bᵢ⁻¹*eᵢeᵢ*Bᵢ⁻¹ / (1. + ωᵢᵗ⁺¹ * Bᵢ⁻¹[i-1,i-1])
                eᵢeᵢ[i-1,i-1] = 0.
            end
            eBe⁻¹ = 1 ./ Bᵢ⁻¹[i,i]
            ωᵢᵗ⁺¹ = (Bᵢ⁻¹*S⁺*Bᵢ⁻¹)[i,i] * eBe⁻¹ * eBe⁻¹ - eBe⁻¹
            Ψ[i] = max(η, (ωᵢᵗ⁺¹ + 1.)*Ψ[i]) # set new parameter
        end

        @debug "Likelihood" iter=c L=L ΔL=abs(L_old - L)
        if abs(L_old - L) < tol
            break
        end
        L_old = L
    end

    return FactorAnalysis(mv, Diagonal(sqrt.(Ψ))*W, Ψ)
end


## interface functions
"""
    fit(FactorAnalysis, X; ...)

Perform factor analysis over the data given in a matrix `X`.
Each column of `X` is an observation.
This method returns an instance of [`FactorAnalysis`](@ref).

**Keyword arguments:**

Let `(d, n) = size(X)` be respectively the input dimension and the number of observations:

- `method`: The choice of methods:
    - `:em`: use EM version of factor analysis
    - `:cm`: use CM version of factor analysis (*default*)
- `maxoutdim`: Maximum output dimension (*default* `d-1`)
- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (*default*)
    - a pre-computed mean vector
- `tol`: Convergence tolerance (*default* `1.0e-6`)
- `maxiter`: Maximum number of iterations (*default* `1000`)
- `η`: Variance low bound (*default* `1.0e-6`)

**Notes:** This function calls [`facm`](@ref) or [`faem`](@ref) internally, depending on the choice of method.
"""
function fit(::Type{FactorAnalysis}, X::AbstractMatrix{T};
             method::Symbol=:cm,
             maxoutdim::Int=size(X,1)-1,
             mean=nothing,
             tol::Real=1.0e-6,   # convergence tolerance
             η = tol,            # variance low bound
             maxiter::Integer=1000) where T<:Real

    d, n = size(X)

    # process mean
    mv = preprocess_mean(X, mean)
    S = covm(X, isempty(mv) ? 0 : mv, 2)
    if method == :em
        M = faem(S, mv, n, maxoutdim=maxoutdim, tol=tol, maxiter=maxiter)
    elseif method == :cm
        M = facm(S, mv, n, maxoutdim=maxoutdim, tol=tol, maxiter=maxiter, η = η)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::FactorAnalysis
end

