using MultivariateStats
using LinearAlgebra
using Test
import Statistics: mean, cov, var
import Random
import SparseArrays
import StatsBase

@testset "Probabilistic PCA" begin

    Random.seed!(34568)

    ## PCA with zero mean

    X = randn(5, 10)
    Y = randn(3, 10)

    W = qr(randn(5, 5)).Q[:, 1:3]
    σ² = 0.1
    M = PPCA(Float64[], W, σ²)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test mean(M) == zeros(5)
    @test loadings(M) == W
    @test var(M) == σ²

    T = inv(W'*W .+ σ²*Matrix(I, 3, 3))*W'
    @test transform(M, X[:,1]) ≈ T * X[:,1]
    @test transform(M, X) ≈ T * X

    R = W*inv(W'W)*(W'W .+ σ²*Matrix(I, 3, 3))
    @test reconstruct(M, Y[:,1]) ≈ R * Y[:,1]
    @test reconstruct(M, Y) ≈ R * Y


    ## PCA with non-zero mean

    mval = rand(5)
    M = PPCA(mval, W, σ²)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test mean(M) == mval
    @test loadings(M) == W
    @test var(M) == σ²

    @test transform(M, X[:,1]) ≈ T * (X[:,1] .- mval)
    @test transform(M, X) ≈ T * (X .- mval)

    @test reconstruct(M, Y[:,1]) ≈ R * Y[:,1] .+ mval
    @test reconstruct(M, Y) ≈ R * Y .+ mval


    ## prepare training data

    d = 5
    n = 1000

    R = collect(qr(randn(d, d)).Q)
    @test R'R ≈ Matrix(I, 5, 5)
    rmul!(R, Diagonal(sqrt.([0.5, 0.3, 0.1, 0.05, 0.05])))

    X = R'randn(5, n) .+ randn(5)
    mval = vec(mean(X, dims=2))
    Z = X .- mval

    M0 = fit(PCA, X; mean=mval, maxoutdim = 4)

    ## ppcaml (default)

    M = fit(PPCA, X)
    P = projection(M)
    W = loadings(M)

    @test indim(M) == 5
    @test outdim(M) == 4
    @test mean(M) == mval
    @test P'P ≈ Matrix(I, 4, 4)
    @test reconstruct(M, transform(M, X)) ≈ reconstruct(M0, transform(M0, X))

    M = fit(PPCA, X; mean=mval)
    @test loadings(M) ≈ W

    M = fit(PPCA, Z; mean=0)
    @test loadings(M) ≈ W

    M = fit(PPCA, X; maxoutdim=3)
    P = projection(M)
    W = loadings(M)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test P'P ≈ Matrix(I, 3, 3)

    # ppcaem

    M = fit(PPCA, X; method=:em)
    P = projection(M)
    W = loadings(M)

    @test indim(M) == 5
    @test outdim(M) == 4
    @test mean(M) == mval
    @test P'P ≈ Matrix(I, 4, 4)
    @test all(isapprox.(reconstruct(M, transform(M, X)), reconstruct(M0, transform(M0, X)), atol=1e-2))

    M = fit(PPCA, X; method=:em, mean=mval)
    @test loadings(M) ≈ W

    M = fit(PPCA, Z; method=:em, mean=0)
    @test loadings(M) ≈ W

    M = fit(PPCA, X; method=:em, maxoutdim=3)
    P = projection(M)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test P'P ≈ Matrix(I, 3, 3)

    @test_throws StatsBase.ConvergenceException fit(PPCA, X; method=:em, maxiter=1)

    # bayespca
    M0 = fit(PCA, X; mean=mval, maxoutdim = 3)

    M = fit(PPCA, X; method=:bayes)
    P = projection(M)
    W = loadings(M)

    @test indim(M) == 5
    @test outdim(M) == 3
    @test mean(M) == mval
    @test P'P ≈ Matrix(I, 3, 3)
    @test reconstruct(M, transform(M, X)) ≈ reconstruct(M0, transform(M0, X))

    M = fit(PPCA, X; method=:bayes, mean=mval)
    @test loadings(M) ≈ W

    M = fit(PPCA, Z; method=:bayes, mean=0)
    @test loadings(M) ≈ W

    M = fit(PPCA, X; method=:em, maxoutdim=2)
    P = projection(M)

    @test indim(M) == 5
    @test outdim(M) == 2
    @test P'P ≈ Matrix(I, 2, 2)

    @test_throws StatsBase.ConvergenceException fit(PPCA, X; method=:bayes, maxiter=1)

    # test that fit works with Float32 values
    X2 = convert(Array{Float32,2}, X)
    # Float32 input, default pratio
    M = fit(PPCA, X2; maxoutdim=3)
    M = fit(PPCA, X2; maxoutdim=3, method=:em)
    M = fit(PPCA, X2; maxoutdim=3, method=:bayes)

    # views
    M = fit(PPCA, view(X2, :, 1:100), maxoutdim=3)
    M = fit(PPCA, view(X2, :, 1:100), maxoutdim=3, method=:em)
    M = fit(PPCA, view(X2, :, 1:100), maxoutdim=3, method=:bayes)
    # sparse
    @test_throws AssertionError fit(PCA, SparseArrays.sprandn(100d, n, 0.6))

end
