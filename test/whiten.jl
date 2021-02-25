using MultivariateStats
using LinearAlgebra, StatsBase, SparseArrays
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
    @test indim(f) == d
    @test outdim(f) == d
    @test size(f) == (d,d)
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

    # mixing types
    X = rand(Float64, 5, 10)
    XX = convert.(Float32, X)

    M = fit(Whitening, X)
    MM = fit(Whitening, XX)

    # mixing types should not error
    transform(M, XX)
    transform(MM, X)

    # type consistency
    @test eltype(mean(M)) == Float64
    @test eltype(mean(MM)) == Float32

    # sparse arrays
    SX = sprand(Float32, d, n, 0.75)
    SM = fit(Whitening, SX; mean=sprand(Float32, 3, 0.75))
    @test transform(SM, SX) isa Matrix{Float32}
end
