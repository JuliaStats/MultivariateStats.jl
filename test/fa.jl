using MultivariateStats
using Base.Test

srand(34568)

## FA with zero mean

X = randn(5, 10)
Y = randn(3, 10)

W = qr(randn(5, 5))[1][:, 1:3]
Ψ = fill(0.1, 5)
M = FactorAnalysis(Float64[], W, Ψ)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == zeros(5)
@test loadings(M) == W
@test var(M) == Ψ

T = inv(I+W'*diagm(1./var(M))*W)*W'*diagm(1./var(M))
@test transform(M, X[:,1]) ≈ T * X[:,1]
@test transform(M, X) ≈ T * X

R = cov(M)*W*inv(W'W)
@test reconstruct(M, Y[:,1]) ≈ R * Y[:,1]
@test reconstruct(M, Y) ≈ R * Y


## PCA with non-zero mean

mv = rand(5)
M = FactorAnalysis(mv, W, Ψ)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == mv
@test loadings(M) == W
@test var(M) == Ψ

@test transform(M, X[:,1]) ≈ T * (X[:,1] .- mv)
@test transform(M, X) ≈ T * (X .- mv)

@test reconstruct(M, Y[:,1]) ≈ R * Y[:,1] .+ mv
@test reconstruct(M, Y) ≈ R * Y .+ mv


## prepare training data

d = 5
n = 1000

R = qr(randn(d, d))[1]
@test R'R ≈ eye(5)
scale!(R, sqrt.([0.5, 0.3, 0.1, 0.05, 0.05]))

X = R'randn(5, n) .+ randn(5)
mv = vec(mean(X, 2))
Z = X .- mv

# facm (default) & faem
fa_methods = [:cm, :em]

fas = FactorAnalysis[]
for method in fa_methods
    M = fit(FactorAnalysis, X, method=method, tot=5000)
    P = projection(M)
    W = loadings(M)
    push!(fas,M)

    @test indim(M) == 5
    @test outdim(M) == 4
    @test mean(M) == mv
    @test P'P ≈ eye(4)
    @test all(isapprox.(cov(M), cov(X, 2), atol=1e-3))

    M = fit(FactorAnalysis, X; mean=mv, method=method)
    @test loadings(M) ≈ W

    M = fit(FactorAnalysis, Z; mean=0, method=method)
    @test loadings(M) ≈ W

    M = fit(FactorAnalysis, X; maxoutdim=3, method=method)
    P = projection(M)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test P'P ≈ eye(3)
end

# compare two algorithms
M1, M2 = fas
@test all(isapprox.(cov(M1), cov(M2), atol=1e-3)) # noise
LL(m, x) = (-size(x,2)/2)*(size(x,1)*log(2π) + log(det(cov(m))) + trace(inv(cov(m))*cov(x,2)))
@test LL(M1, X) ≈ LL(M2, X) # log likelihood

# test that fit works with Float32 values
X2 = convert(Array{Float32,2}, X)
# Float32 input
M = fit(FactorAnalysis, X2; method=:cm, maxoutdim=3)
M = fit(FactorAnalysis, X2; method=:em, maxoutdim=3)
