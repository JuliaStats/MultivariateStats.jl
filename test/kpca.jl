using MultivariateStats
using LinearAlgebra
using Test
import SparseArrays
import Statistics: mean, cov
import Random


@testset "Kernel PCA" begin

    Random.seed!(34568)

    ## data
    n = 10
    d = 5
    X = randn(d, n)

    # step-by-step kernel centralization
    for K in [
        reshape(1.:12., 3, 4),
        reshape(1.:9., 3, 3),
        reshape(1.:12., 4, 3),
        rand(n,d),
        rand(d,d),
        rand(d,n) ]

        x, y = size(K)
        I1 = ones(x,x)/x
        I2 = ones(y,y)/y
        Z = K - I1*K - K*I2 + I1*K*I2

        KC = fit(MultivariateStats.KernelCenter, K)
        @test all(isapprox.(Z, MultivariateStats.transform!(KC, copy(K))))
    end

    # kernel calculations
    ker1  = (x,y)->x'y
    ker2  = (x,y)->norm(x-y)

    K = MultivariateStats.pairwise(ker1, X)
    @test size(K) == (n,n)
    @test K[1,2] == K[2,1]

    K = MultivariateStats.pairwise(ker1, X, X.+1)
    @test size(K) == (n,n)
    @test K[1,2] != K[2,1]

    K = MultivariateStats.pairwise(ker2, X, X[:,1:2])
    @test size(K) == (n, 2)
    @test K[1,1] == 0
    @test K[3,2] == norm(X[:,3] - X[:,2])

    K = MultivariateStats.pairwise(ker2, X[:,1:3], X)
    @test size(K) == (3, n)
    @test K[1,1] == 0
    @test K[3,2] == norm(X[:,2] - X[:,3])

    K = similar(X, n, n)
    MultivariateStats.pairwise!(K, ker2, X)
    @test size(K) == (n, n)
    @test K[1,1] == 0
    @test K[2,1] == norm(X[:,2] - X[:,1])

    KC = fit(MultivariateStats.KernelCenter, K)
    Iₙ = ones(n,n)/n
    @test MultivariateStats.transform!(KC, copy(K)) ≈ K - Iₙ*K - K*Iₙ + Iₙ*K*Iₙ

    K = MultivariateStats.pairwise(ker2, X, X[:,1])
    @test size(K) == (n, 1)
    @test K[1,1] == 0
    @test K[2,1] == norm(X[:,2] - X[:,1])

    KC = fit(MultivariateStats.KernelCenter, K)
    @test all(isapprox.(MultivariateStats.transform!(KC, copy(K)), 0.0, atol=10e-7))

    ## check different parameters
    X = randn(d, n)
    M = fit(KernelPCA, X, maxoutdim=d)
    M2 = fit(PCA, X, method=:cov, pratio=1.0)
    @test indim(M) == d
    @test outdim(M) == d
    @test abs.(transform(M)) ≈ abs.(transform(M2, X))
    @test abs.(transform(M, X)) ≈ abs.(transform(M2, X))
    @test abs.(transform(M, X[:,1])) ≈ abs.(transform(M2, X[:,1]))

    M = fit(KernelPCA, X, maxoutdim=3, solver=:eigs)
    M2 = fit(PCA, X, method=:cov, maxoutdim=3)
    @test indim(M) == d
    @test outdim(M) == 3
    @test abs.(transform(M, X)) ≈ abs.(transform(M2, X))
    @test abs.(transform(M, X[:,1])) ≈ abs.(transform(M2, X[:,1]))

    # issue #44
    Y = randn(d, 2*n)
    @test size(transform(M, Y)) == size(transform(M2, Y))

    # reconstruction
    @test_throws ArgumentError reconstruct(M, X)
    M = fit(KernelPCA, X, inverse=true)
    @test all(isapprox.(reconstruct(M, transform(M)), X, atol=0.75))

    # use rbf kernel
    γ = 10.
    rbf=(x,y)->exp(-γ*norm(x-y)^2.0)
    M = fit(KernelPCA, X, kernel=rbf)
    @test indim(M) == d
    @test outdim(M) == d

    # use precomputed kernel
    K = MultivariateStats.pairwise((x,y)->x'*y, X)
    @test_throws AssertionError fit(KernelPCA, rand(1,10), kernel=nothing) # symmetric kernel
    M = fit(KernelPCA, K, maxoutdim = 5, kernel=nothing, inverse=true) # use precomputed kernel
    M2 = fit(PCA, X, method=:cov, pratio=1.0)
    @test_throws ArgumentError reconstruct(M, X) # no reconstruction for precomputed kernel
    @test abs.(transform(M)) ≈ abs.(transform(M2, X))

    @test_throws TypeError fit(KernelPCA, rand(1,10), kernel=1)

    # different types
    X = randn(Float64, d, n)
    XX = convert.(Float32, X)

    M = fit(KernelPCA, X ; inverse=true)
    MM = fit(KernelPCA, XX ; inverse=true)

    Y = randn(Float64, outdim(M))
    YY = convert.(Float32, Y)

    @test indim(MM) == d
    @test outdim(MM) == d
    @test eltype(transform(MM, X[:,1])) == Float32

    for func in (projection, principalvars)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end

    # mixing types should not error
    transform(M, XX)
    transform(MM, X)
    reconstruct(M, YY)
    reconstruct(MM, Y)

    ## fit a sparse matrix
    X = SparseArrays.sprandn(100d, n, 0.6)
    M = fit(KernelPCA, X, maxoutdim=3, solver=:eigs)
    @test indim(M) == 100d
    @test outdim(M) == 3
end
