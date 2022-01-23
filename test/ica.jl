using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
using Statistics: mean, cov
using StatsBase: ConvergenceException

@testset "ICA" begin

    rng = StableRNG(15678)

    function generatetestdata(rng, n, k, m)
        t = range(0.0, step=10.0, length=n)
        s1 = sin.(t * 2)
        s2 = s2 = 1.0 .- 2.0 * Bool[isodd(floor(Int, x / 3)) for x in t]
        s3 = Float64[mod(x, 5.0) for x in t]

        s1 += 0.1 * randn(rng, n)
        s2 += 0.1 * randn(rng, n)
        s3 += 0.1 * randn(rng, n)

        S = hcat(s1, s2, s3)'
        @assert size(S) == (k, n)

        A = randn(rng, m, k)

        X = A * S
        mv = vec(mean(X, dims=2))
        @assert size(X) == (m, n)
        C = cov(X, dims=2)
        return X, mv, C
    end

    @testset "Auxiliary" begin

        f = MultivariateStats.Tanh(1.0)
        u, v = ones(1,1), zeros(1)
        MultivariateStats.update!(f, u, v)
        @test u[] ≈ 0.7615941559557649
        @test v[] ≈ 0.41997434161402614

        f = MultivariateStats.Tanh(1.0f0)
        u, v = ones(Float32,1,1), zeros(Float32,1)
        MultivariateStats.update!(f, u, v)
        @test u[] ≈ 0.7615942f0
        @test v[] ≈ 0.41997433f0

        f = MultivariateStats.Gaus()
        u, v = ones(1,1), ones(1)
        MultivariateStats.update!(f, u, v)
        @test u[] ≈ 0.6065306597126334
        @test v[] ≈ 0.0

        u, v = ones(Float32,1,1), ones(Float32,1)
        MultivariateStats.update!(f, u, v)
        @test u[] ≈ 0.60653067f0
        @test v[] ≈ 0.0f0

    end

    ## data
    @testset "Algorithm" begin

        # sources
        n = 1000
        k = 3
        m = 8
        X, μ, C = generatetestdata(rng, n, k, m)

        # FastICA

        M = fit(ICA, X, k; do_whiten=false, tol=Inf)
        @test isa(M, ICA)
        @test size(M) == (m,k)
        @test mean(M) == μ
        W = M.W
        @test predict(M, X) ≈ W' * (X .- μ)
        @test W'W ≈ Matrix(I, k, k)

        M = fit(ICA, X, k; do_whiten=true, tol=Inf)
        @test isa(M, ICA)
        @test size(M) == (m,k)
        @test mean(M) == μ
        W = M.W
        @test W'C * W ≈ Matrix(I, k, k)

        @test_throws ConvergenceException fit(ICA, X, k; do_whiten=true, tol=1e-8, maxiter=2)

        # Use data of different type
        XX = convert(Matrix{Float32}, X)

        MM = fit(ICA, XX, k; do_whiten=true, tol=Inf)
        @test eltype(mean(MM)) == Float32
        @test eltype(MM.W) == Float32

        MM = fit(ICA, XX, k; do_whiten=false, tol=Inf)
        @test isa(MM, ICA)
        @test eltype(mean(MM)) == Float32
        @test eltype(MM.W) == Float32
        W = MM.W
        @test predict(MM, X) ≈ W' * convert(Matrix{Float32}, (X .- μ))
        @test predict(M, XX) ≈ M.W' * (X .- μ) atol=1e-4
        @test W'W ≈ Matrix{Float32}(I, k, k)

        # input as view
        M = fit(ICA, view(XX, :, 1:400), k; do_whiten=true, tol=Inf)
        @test eltype(mean(MM)) == Float32
        @test eltype(MM.W) == Float32
    end

end
