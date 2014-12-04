
type FactorAnalysis
  Ψ::Vector{Float64}
  F::Matrix{Float64}
end

n_components(fa::FactorAnalysis) = size(fa.F, 2)

const SMALL = 1e-12

function fit(::Type{FactorAnalysis}, V::Matrix{Float64}, n_components::Int, tol::Float64=1e-2, max_iter::Int=1000, Ψ::Vector{Float64} = ones(Float64, size(V,1)))
  Vbar = mean(V, 2)
  σ2 = vec(var(V, 2))
  D, N = size(V)
  X = V .- Vbar
  L_old = -Inf
  L_const = D + log(2π) + n_components
  L = -Inf
  F = zeros(D, n_components)
  k = 0
  for k = 1:max_iter
    Ψ_sqrt = diagm(sqrt(Ψ) + SMALL)
    X_tilde = Ψ_sqrt^-1 * X/(sqrt(N))
    U, Λ_tilde, W_tilde = svd(X_tilde, thin=true)
    Λ = Λ_tilde.^2
    U_H = U[:,1:n_components]
    Λ_H = Λ[1:n_components]
    # Factor update
    F = Ψ_sqrt * U_H * diagm(max(Λ_H .- 1, 0).^0.5)
    # Log likelihood
    L_old = L
    L = L_const + log(prod(2π*Ψ))
    for i = 1:n_components
      L += log(Λ[i])
    end
    for i = n_components+1:D
      L += Λ[i]
    end
    L *= -N/2
    Ψ = max(diag(diagm(σ2) - F*F'), SMALL)

    if L - L_old < tol
      break
    end
  end
  if k == max_iter
    info("Factor analysis did not converge in $max_iter iterations.")
  end
  FactorAnalysis(Ψ, F)
end


