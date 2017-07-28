# Factor Analysis

"""Factor Analysis type"""
immutable FactorAnalysis{T<:AbstractFloat}
    mean::Vector{T}       # sample mean: of length d (mean can be empty, which indicates zero mean)
    W::Matrix{T}          # factor loadings matrix: of size d x p
    Ψ::Vector{T}          # noise covariance: diagonal of size d x d
end

indim(M::FactorAnalysis) = size(M.W, 1)
outdim(M::FactorAnalysis) = size(M.W, 2)

Base.mean(M::FactorAnalysis) = fullmean(indim(M), M.mean)
projection(M::FactorAnalysis) = svdfact(M.W)[:U] # recover principle components from the weight matrix
Base.cov(M::FactorAnalysis) = M.W*M.W'+ Diagonal(M.Ψ)
Base.var(M::FactorAnalysis) = M.Ψ
loadings(M::FactorAnalysis) = M.W

## use

function transform{T<:AbstractFloat}(m::FactorAnalysis{T}, x::AbstractVecOrMat{T})
    xn = centralize(x, mean(m))
    W = m.W
    WᵀΨ⁻¹ = W'*diagm(1./m.Ψ)  # (q x d) * (d x d) = (q x d)
    return inv(I+WᵀΨ⁻¹*W)*(WᵀΨ⁻¹*xn)  # (q x q) * (q x d) * (d x 1) = (q x 1)
end

function reconstruct{T<:AbstractFloat}(m::FactorAnalysis{T}, z::AbstractVecOrMat{T})
    W  = m.W
    # ΣW(W'W)⁻¹z+μ = ΣW(W'W)⁻¹W'Σ⁻¹(x-μ)+μ = Σ(WW⁻¹)((W')⁻¹W')Σ⁻¹(x-μ)+μ = ΣΣ⁻¹(x-μ)+μ = (x-μ)+μ = x
    return cov(m)*W*inv(W'W)*z .+ mean(m)
end

## show

function Base.show(io::IO, M::FactorAnalysis)
    print(io, "Factor Analysis(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## core algorithms

""" Expectation-maximization (EM) algorithms for maximum likelihood factor analysis.

    Rubin, Donald B., and Dorothy T. Thayer. "EM algorithms for ML factor analysis." Psychometrika 47.1 (1982): 69-76.
"""
function faem{T<:AbstractFloat}(S::DenseMatrix{T}, mv::Vector{T}, n::Int;
             maxoutdim::Int=size(X,1)-1,
             tol::Real=1.0e-6,   # convergence tolerance
             tot::Integer=1000)  # maximum number of iterations

    d = size(S,1)
    q = maxoutdim
    W = eye(T,d,q)
    Ψ = fill(T(0.01),d)

    L_old = 0.
    for c in 1:tot
        # EM-steps
        Ψ⁻¹W = diagm(1./Ψ)*W
        SΨ⁻¹W = S*Ψ⁻¹W
        H = SΨ⁻¹W*inv(I + W'*Ψ⁻¹W)
        W⁺ = SΨ⁻¹W*inv(I + H'*Ψ⁻¹W)
        Ψ⁺ = diag(S - H*(W⁺)')
        # new parameters
        W = W⁺
        Ψ = Ψ⁺
        # log likelihood
        Ψ⁻¹ = diagm(1./Ψ)
        WᵀΨ⁻¹ = W'*Ψ⁻¹
        detΣ = prod(Ψ)*det(I + WᵀΨ⁻¹*W)
        Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹*W*inv(I + WᵀΨ⁻¹*W)*WᵀΨ⁻¹
        L = (-n/2)*(d*log(2π) + log(detΣ) + trace(Σ⁻¹*S))
        # println("$c] ΔL: $(abs(L_old - L)), L: $L")
        if abs(L_old - L) < tol
            break
        end
        L_old = L
    end

    return FactorAnalysis(mv, W, Ψ)
end

""" Fast conditional maximization (CM) algorithm for factor analysis.

    Zhao, J-H., Philip LH Yu, and Qibao Jiang. "ML estimation for factor analysis: EM or non-EM?." Statistics and computing 18.2 (2008): 109-123.
"""
function facm{T<:AbstractFloat}(S::DenseMatrix{T}, mv::Vector{T}, n::Int;
             maxoutdim::Int=size(X,1)-1,
             tol::Real=1.0e-6,   # convergence tolerance
             tot::Integer=1000,  # maximum number of iterations
             η = tol)            # variance low bound
    d = size(S,1)

    q = maxoutdim
    W = eye(T,d,q)
    Ψ = fill(T(0.01),d)
    V = zeros(T,q)
    eᵢeᵢ = zeros(T,d,d)
    Bᵢ⁻¹ = zeros(T,d,d)
    addconst = d*log(2π)

    L_old = 0.
    for c in 1:tot
        # CM-step 1
        Ψ⁻ʰ = Diagonal(1./sqrt.(Ψ))
        S⁺ = Ψ⁻ʰ*S*Ψ⁻ʰ

        F = eigfact(S⁺)
        λ = real(F[:values])
        ord = sortperm(λ, rev=true)
        λ = λ[ord]

        q′ = λ[q] > 0 ? q : findlast(λ .> 0)

        λq = 0.0
        @inbounds for i in 1:q′
            λq += log(λ[i]) - λ[i] + 1.
            V[i] = sqrt(λ[i] - 1.)
        end

        L = (addconst + log(prod(Ψ)) + trace(S⁺) + λq)*(-n/2)

        U = convert(Matrix{T}, F[:vectors][:,ord[1:q′]])
        W = U*diagm(V[1:q′]) # set new parameter

        # CM-step 2
        @inbounds for i in 1:q′
            V[i] = 1./λ[i] - 1.
        end
        eBe⁻¹ = ωᵢᵗ⁺¹ = 0.0
        for i in 1:d
            if i == 1
                Bᵢ⁻¹ = U*diagm(V[1:q′])*U' + I
            else
                eᵢeᵢ[i-1,i-1] = 1.
                Bᵢ⁻¹ = Bᵢ⁻¹ -  ωᵢᵗ⁺¹*Bᵢ⁻¹*eᵢeᵢ*Bᵢ⁻¹ / (1. + ωᵢᵗ⁺¹ * Bᵢ⁻¹[i-1,i-1])
                eᵢeᵢ[i-1,i-1] = 0.
            end
            eBe⁻¹ = 1./Bᵢ⁻¹[i,i]
            ωᵢᵗ⁺¹ = (Bᵢ⁻¹*S⁺*Bᵢ⁻¹)[i,i] * eBe⁻¹ * eBe⁻¹ - eBe⁻¹
            Ψ[i] = max(η, (ωᵢᵗ⁺¹ + 1.)*Ψ[i]) # set new parameter
        end

        # println("$c] ΔL: $(abs(L_old - L)), L: $L")
        if abs(L_old - L) < tol
            break
        end
        L_old = L
    end

    return FactorAnalysis(mv, Diagonal(sqrt.(Ψ))*W, Ψ)
end


## interface functions
function fit{T<:AbstractFloat}(::Type{FactorAnalysis}, X::DenseMatrix{T};
             method::Symbol=:cm,
             maxoutdim::Int=size(X,1)-1,
             mean=nothing,
             tol::Real=1.0e-6,   # convergence tolerance
             tot::Integer=1000,  # maximum number of iterations
             η = tol)            # variance low bound

    d, n = size(X)

    # process mean
    mv = preprocess_mean(X, mean)
    S = Base.covm(X, isempty(mv) ? 0 : mv, 2)
    if method == :em
        M = faem(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot)
    elseif method == :cm
        M = facm(S, mv, n, maxoutdim=maxoutdim, tol=tol, tot=tot, η = η)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end

    return M::FactorAnalysis
end
