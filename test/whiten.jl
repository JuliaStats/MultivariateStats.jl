using MultivariateStats
using LinearAlgebra, StatsBase, SparseArrays
using Test
using StableRNGs
import Statistics: mean, cov

@testset "Whitening" begin

    rng = StableRNG(34568)

    ## data

    d = 3
    n = 5

    X = rand(rng, d, n)
    mval = vec(mean(X, dims=2))
    C = cov(X, dims=2)
    C0 = copy(C)

    emax = maximum(eigvals(C))
    rc = 0.01
    Cr = C + (emax * rc) * Matrix(I, d, d)

    ## cov_whitening

    cf = cholesky(Hermitian(C, :U))
    W = cov_whitening(cf)
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)

    cf = cholesky(Hermitian(C, :L))
    W = cov_whitening(cf)
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)

    W = cov_whitening(C)
    @test C == C0
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)

    W = cov_whitening(C, 0.0)
    @test C == C0
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)

    W = cov_whitening(C, rc)
    @test istriu(W)
    @test W'Cr * W ≈ Matrix(I, d, d)


    # Whitening from data

    f = fit(Whitening, X, dims=2)
    W = f.W
    @test isa(f, Whitening{Float64})
    @test mean(f) === f.mean
    @test length(f) == d
    @test size(f) == (d,d)
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)
    @test MultivariateStats.transform(f, X) ≈ W' * (X .- f.mean)

    f = fit(Whitening, X; regcoef=rc, dims=2)
    W = f.W
    @test W'Cr * W ≈ Matrix(I, d, d)

    f = fit(Whitening, X; mean=mval, dims=2)
    W = f.W
    @test W'C * W ≈ Matrix(I, d, d)

    f = fit(Whitening, X; mean=0, dims=2)
    Cx = (X * X') / (n - 1)
    W = f.W
    @test W'Cx * W ≈ Matrix(I, d, d)

    # invsqrtm

    R = invsqrtm(C)
    @test C == C0
    @test R ≈ inv(sqrt(C))

    # mixing types
    X = rand(rng, Float64, 5, 10)
    XX = convert.(Float32, X)

    M = fit(Whitening, X, dims=2)
    MM = fit(Whitening, XX, dims=2)

    # mixing types should not error
    MultivariateStats.transform(M, XX)
    MultivariateStats.transform(MM, X)

    # type consistency
    @test eltype(mean(M)) == Float64
    @test eltype(mean(MM)) == Float32

    # sparse arrays
    SX = sprand(rng, Float32, d, n, 0.75)
    SM = fit(Whitening, SX; mean=sprand(rng, Float32, 3, 0.75), dims=2)
    Y = MultivariateStats.transform(SM, SX)
    @test eltype(Y) == Float32

    # different dimensions
    @test_throws DomainError fit(Whitening, X'; dims=3)
    M1 = fit(Whitening, X'; dims=1)
    M2 = fit(Whitening, X; dims=2)
    @test M1.W == M2.W
    @test_throws DimensionMismatch MultivariateStats.transform(M1, rand(rng, 6,4))
    @test_throws DimensionMismatch MultivariateStats.transform(M2, rand(rng, 4,6))
    Y1 = MultivariateStats.transform(M1,X')
    Y2 = MultivariateStats.transform(M2,X)
    @test Y1' == Y2
    @test_throws DimensionMismatch MultivariateStats.transform(M1, rand(rng, 7))
    V1 = MultivariateStats.transform(M1,X[:,1])
    V2 = MultivariateStats.transform(M2,X[:,1])
    @test V1 == V2
end
