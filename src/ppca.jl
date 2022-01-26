# Probabilistic Principal Component Analysis

"""
This type contains probabilistic PCA model parameters.
"""
struct PPCA{T<:Real} <: LatentVariableDimensionalityReduction
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    W::Matrix{T}          # weight matrix: of size d x p
    σ²::T                 # residual variance
end

## properties
"""
    size(M::PPCA)

Returns a tuple with values of the input dimension ``d``, *i.e* the dimension of
the observation space, and the output dimension ``p``, *i.e* the dimension of
the principal subspace.
"""
size(M::PPCA) = size(M.W)

"""
    mean(M::PPCA)

Get the mean vector (of length ``d``).
"""
mean(M::PPCA) = fullmean(size(M)[1], M.mean)

"""
    projection(M::PPCA)

Returns the projection matrix (of size ``(d, p)``). Each column of the projection
matrix corresponds to a principal component.

The principal components are arranged in descending order of the corresponding variances.

"""
projection(M::PPCA) = svd(M.W).U # recover principle components from the weight matrix

"""
    var(M::PPCA)

Returns the total residual variance of the model `M`.
"""
var(M::PPCA) = M.σ²

"""
    loadings(M::PPCA)

Returns the factor loadings matrix (of size ``(d, p)``) of the model `M`.
"""
loadings(M::PPCA) = M.W

"""
    cov(M::PPCA)

Returns the covariance of the model `M`.
"""
cov(M::PPCA) = M.W'*M.W + M.σ²*I

## use
"""
    predict(M::PPCA, x)

Transform observations `x` into latent variables. Here, `x` can be either a vector
of length `d` or a matrix where each column is an observation.
"""
function predict(M::PPCA, x::AbstractVecOrMat{<:Real})
    xn = centralize(x, M.mean)
    n = size(M)[2]
    return inv(cov(M))*M.W'*xn
end

"""
    reconstruct(M::PPCA, z)

Approximately reconstruct observations from the latent variable given in `z`.
Here, `z` can be either a vector of length `p` or a matrix where each column gives
the latent variables for an observation.
"""
function reconstruct(M::PPCA, z::AbstractVecOrMat{<:Real})
    W  = M.W
    WTW = W'W
    n = size(M)[2]
    C  = WTW + var(M) * I
    return W*inv(WTW)*C*z .+ mean(M)
end

## show

function Base.show(io::IO, M::PPCA)
    i, o = size(M)
    print(io, "Probabilistic PCA(indim = $i, outdim = $o, σ² = $(var(M)))")
end

## core algorithms
"""
    ppcaml(Z, mean; ...)

Compute probabilistic PCA using on maximum likelihood formulation for a centralized
sample matrix `Z`.

*Parameters*:
- `Z`: a centralized samples matrix
- `mean`: The mean vector of the **original** samples, which can be a vector of
length `d`, or an empty vector indicating a zero mean.

Returns the resultant [`PPCA`](@ref) model.

**Note:** This function accepts two keyword arguments: `maxoutdim` and `tol`.
"""
function ppcaml(Z::AbstractMatrix{T}, mean::Vector{T};
                tol::Real=1.0e-6, # convergence tolerance
                maxoutdim::Int=size(Z,1)-1) where {T<:Real}

    check_pcaparams(size(Z,1), mean, maxoutdim, 1.)

    d, n = size(Z)

    # SVD decomposition
    Svd = svd(Z)
    λ = Svd.S
    ord = sortperm(λ; rev=true)
    V = abs2.(λ[ord]) ./ (n-1)

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

"""
    ppcaem(S, mean, n; ...)

Compute probabilistic PCA based on expectation-maximization algorithm for a given sample covariance matrix `S`.

*Parameters*:
- `S`: The sample covariance matrix.
- `mean`: The mean vector of original samples, which can be a vector of length `d`,
or an empty vector indicating a zero mean.
- `n`: The number of observations.

Returns the resultant [`PPCA`](@ref) model.

**Note:** This function accepts three keyword arguments: `maxoutdim`, `tol`, and `maxiter`.
"""
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


"""
    bayespca(S, mean, n; ...)

Compute probabilistic PCA using a Bayesian algorithm for a given sample covariance matrix `S`.

*Parameters*:
- `S`: The sample covariance matrix.
- `mean`: The mean vector of original samples, which can be a vector of length `d`,
or an empty vector indicating a zero mean.
- `n`: The number of observations.

Returns the resultant [`PPCA`](@ref) model.

**Notes:**
- This function accepts three keyword arguments: `maxoutdim`, `tol`, and `maxiter`.
- Function uses the `maxoutdim` parameter as an upper boundary when it automatically
determines the latent space dimensionality.
"""
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
"""

    fit(PPCA, X; ...)

Perform probabilistic PCA over the data given in a matrix `X`.
Each column of `X` is an observation. This method returns an instance of [`PPCA`](@ref).

**Keyword arguments:**

Let `(d, n) = size(X)` be respectively the input dimension and the number of observations:

- `method`: The choice of methods:
    - `:ml`: use maximum likelihood version of probabilistic PCA (*default*)
    - `:em`: use EM version of probabilistic PCA
    - `:bayes`: use Bayesian PCA
- `maxoutdim`: Maximum output dimension (*default* `d-1`)
- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (*default*)
    - a pre-computed mean vector
- `tol`: Convergence tolerance (*default* `1.0e-6`)
- `maxiter`: Maximum number of iterations (*default* `1000`)

**Notes:** This function calls [`ppcaml`](@ref), [`ppcaem`](@ref) or
[`bayespca`](@ref) internally, depending on the choice of method.
"""
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

