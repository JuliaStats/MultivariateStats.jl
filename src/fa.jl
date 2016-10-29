using Debug

type FactorAnalysis
  Ψ::Vector{Float64}
  F::Matrix{Float64}
  Vbar::Vector{Float64}
end

n_components(fa::FactorAnalysis) = size(fa.F, 2)

const SMALL = 1e-12

@debug function fit(::Type{FactorAnalysis}, V::Matrix{Float64}, n_components::Int, tol::Float64=1e-2, max_iter::Int=1000, Ψ::Vector{Float64} = ones(Float64, size(V,1)))
  Vbar = vec(mean(V, 2))
  σ2 = vec(var(V, 2, corrected=false)) # corrected = false used to get same results as scikit-learn
  D, N = size(V)
  X = centralize(V, Vbar)
  L_old = -Inf
  L = -Inf
  F = zeros(D, n_components)
  k = 0
  for k = 1:max_iter
    Ψ_sqrt = sqrt(Ψ) + SMALL
    X_tilde = 1./Ψ_sqrt .* X/(sqrt(N))
    U, Λ_tilde, _ = svd(X_tilde, thin=true)
    Λ = Λ_tilde.^2
    U_H = U[:,1:n_components]
    Λ_H = Λ[1:n_components]
    # Factor update
    @bp
    F = Ψ_sqrt .* U_H * diagm(sqrt(max(Λ_H .- 1, 0)))
    # Log likelihood
    L_old = L
    L = n_components + log(prod(2π*Ψ))
    for i = 1:n_components
      @inbounds L += log(Λ[i])
    end
    for i = n_components+1:D
      @inbounds L += Λ[i]
    end
    L *= -N/2
    # Noise update
    Ψ = vec(max(σ2 - sum(F.^2, 2), SMALL))

    if L - L_old < tol
      info("Factor analysis converged, log-likelihood = $L")
      break
    end
  end
  if k == max_iter
    info("Factor analysis did not converge in $max_iter iterations.")
  end
  FactorAnalysis(Ψ, F, Vbar), L
end

function transform(fa::FactorAnalysis, data::Matrix{Float64})
# FIXME
  X = centralize(data, fa.Vbar)
  F_Ψ = fa.F ./ fa.Ψ
  cov_z = inv(eye(n_components(fa)) .+ F_Ψ' * fa.Ψ)
  cov_z' * F_Ψ' * X
end

