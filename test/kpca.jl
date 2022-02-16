using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
import SparseArrays
import Statistics: mean, cov


@testset "Kernel PCA" begin

    rng = StableRNG(34568)

    ## data
    n = 10
    d = 5
    X = randn(rng, d, n)

    # step-by-step kernel centralization
    for K in [
        reshape(1.:12., 3, 4),
        reshape(1.:9., 3, 3),
        reshape(1.:12., 4, 3),
        rand(rng, n,d),
        rand(rng, d,d),
        rand(rng, d,n) ]

        x, y = size(K)
        I1 = ones(x,x)/x
        I2 = ones(y,y)/y
        Z = K - I1*K - K*I2 + I1*K*I2

        KC = fit(MultivariateStats.KernelCenter, K)
        @test all(Z .≈ MultivariateStats.transform!(KC, copy(K)))
    end

    # kernel calculations
    ker  = (x,y)->norm(x-y)

    K = similar(X, n, n)
    MultivariateStats.pairwise!(ker, K, eachcol(X))
    @test size(K) == (n, n)
    @test K[1,1] == 0
    @test K[2,1] == norm(X[:,2] - X[:,1])

    KC = fit(MultivariateStats.KernelCenter, K)
    Iₙ = ones(n,n)/n
    @test MultivariateStats.transform!(KC, copy(K)) ≈ K - Iₙ*K - K*Iₙ + Iₙ*K*Iₙ

    K = MultivariateStats.pairwise(ker, eachcol(X), eachcol(X[:,1]))
    @test size(K) == (n, 1)
    @test K[1,1] == 0
    @test K[2,1] == norm(X[:,2] - X[:,1])

    KC = fit(MultivariateStats.KernelCenter, K)
    @test all(isapprox.(MultivariateStats.transform!(KC, copy(K)), 0.0, atol=10e-7))

    ## check different parameters
    X = randn(rng, d, n)
    M = fit(KernelPCA, X, maxoutdim=d)
    M2 = fit(PCA, X, method=:cov, pratio=1.0)
    @test size(M) == (d,d)
    @test abs.(predict(M)) ≈ abs.(predict(M2, X))
    @test abs.(predict(M, X)) ≈ abs.(predict(M2, X))
    @test abs.(predict(M, X[:,1])) ≈ abs.(predict(M2, X[:,1]))

    M = fit(KernelPCA, X, maxoutdim=3, solver=:eigs)
    M2 = fit(PCA, X, method=:cov, maxoutdim=3)
    @test size(M)[1] == d
    @test size(M)[2] == 3
    @test abs.(predict(M, X)) ≈ abs.(predict(M2, X))
    @test abs.(predict(M, X[:,1])) ≈ abs.(predict(M2, X[:,1]))

    # issue #44
    Y = randn(rng, d, 2*n)
    @test size(predict(M, Y)) == size(predict(M2, Y))

    # reconstruction
    @test_throws ArgumentError reconstruct(M, X)
    M = fit(KernelPCA, X, inverse=true)
    @test all(isapprox.(reconstruct(M, predict(M)), X, atol=0.75))

    # use RBF kernel
    γ = 10.
    rbf=(x,y)->exp(-γ*norm(x-y)^2.0)
    M = fit(KernelPCA, X, kernel=rbf)
    @test size(M) == (d,d)

    # use precomputed kernel
    K = MultivariateStats.pairwise((x,y)->x'*y, eachcol(X), symmetric=true)
    @test_throws AssertionError fit(KernelPCA, rand(rng, 1,10), kernel=nothing) # symmetric kernel
    M = fit(KernelPCA, K, maxoutdim = 5, kernel=nothing, inverse=true) # use precomputed kernel
    M2 = fit(PCA, X, method=:cov, pratio=1.0)
    @test_throws ArgumentError reconstruct(M, X) # no reconstruction for precomputed kernel
    @test abs.(predict(M)) ≈ abs.(predict(M2, X))

    @test_throws TypeError fit(KernelPCA, rand(rng, 1,10), kernel=1)

    # different types
    X = randn(rng, Float64, d, n)
    XX = convert.(Float32, X)

    M = fit(KernelPCA, X ; inverse=true)
    MM = fit(KernelPCA, XX ; inverse=true)

    Y = randn(rng, Float64, size(M)[2])
    YY = convert.(Float32, Y)

    @test size(MM) == (d,d)
    @test eltype(predict(MM, XX[:,1])) == Float32

    for func in (projection, eigvals)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end

    # mixing types should not error
    predict(M, XX)
    predict(MM, X)
    reconstruct(M, YY)
    reconstruct(MM, Y)

    ## fit a sparse matrix
    X = SparseArrays.sprandn(rng, 100d, n, 0.6)
    M = fit(KernelPCA, X, maxoutdim=3, solver=:eigs)
    @test size(M)[1] == 100d
    @test size(M)[2] == 3
end
