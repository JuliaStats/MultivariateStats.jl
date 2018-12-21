using MultivariateStats
using LinearAlgebra
using Test
import Statistics: mean, cov
import Random

@testset "Whitening" begin

    Random.seed!(34568)

    ## data

    d = 3
    n = 5

    X = rand(d, n)
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

    f = fit(Whitening, X)
    W = f.W
    @test isa(f, Whitening{Float64})
    @test mean(f) === f.mean
    @test istriu(W)
    @test W'C * W ≈ Matrix(I, d, d)
    @test transform(f, X) ≈ W' * (X .- f.mean)

    f = fit(Whitening, X; regcoef=rc)
    W = f.W
    @test W'Cr * W ≈ Matrix(I, d, d)

    f = fit(Whitening, X; mean=mval)
    W = f.W
    @test W'C * W ≈ Matrix(I, d, d)

    f = fit(Whitening, X; mean=0)
    Cx = (X * X') / (n - 1)
    W = f.W
    @test W'Cx * W ≈ Matrix(I, d, d)

    # invsqrtm

    R = invsqrtm(C)
    @test C == C0
    @test R ≈ inv(sqrt(C))

end
