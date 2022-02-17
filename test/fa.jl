using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
import Statistics: mean, cov, var

@testset "Factor Analysis" begin

    rng = StableRNG(34568)

    ## FA with zero mean

    X = randn(rng, 5, 10)
    Y = randn(rng, 3, 10)

    W = qr(randn(rng, 5, 5)).Q[:, 1:3]
    Ψ = fill(0.1, 5)
    M = FactorAnalysis(Float64[], W, Ψ)

    @test size(M) == (5, 3)
    @test mean(M) == zeros(5)
    @test loadings(M) == W
    @test var(M) == Ψ

    T = inv(I+W'*diagm(0 => 1 ./ var(M))*W)*W'*diagm(0 => 1 ./ var(M))
    @test predict(M, X[:,1]) ≈ T * X[:,1]
    @test predict(M, X) ≈ T * X

    R = cov(M)*W*inv(W'W)
    @test reconstruct(M, Y[:,1]) ≈ R * Y[:,1]
    @test reconstruct(M, Y) ≈ R * Y


    ## PCA with non-zero mean

    mval = rand(rng, 5)
    M = FactorAnalysis(mval, W, Ψ)

    @test size(M) == (5, 3)
    @test mean(M) == mval
    @test loadings(M) == W
    @test var(M) == Ψ

    @test predict(M, X[:,1]) ≈ T * (X[:,1] .- mval)
    @test predict(M, X) ≈ T * (X .- mval)

    @test reconstruct(M, Y[:,1]) ≈ R * Y[:,1] .+ mval
    @test reconstruct(M, Y) ≈ R * Y .+ mval


    ## prepare training data

    d = 5
    n = 1000

    R = collect(qr(randn(rng, d, d)).Q)
    @test R'R ≈ Matrix(I, 5, 5)
    rmul!(R, Diagonal(sqrt.([0.5, 0.3, 0.1, 0.05, 0.05])))

    X = R'randn(rng, 5, n) .+ randn(rng, 5)
    mval = vec(mean(X, dims=2))
    Z = X .- mval

    # facm (default) & faem
    fa_methods = [:cm, :em]

    fas = FactorAnalysis[]
    for method in fa_methods
        let M = fit(FactorAnalysis, X, method=method, maxiter=5000),
            P = projection(M),
            W = loadings(M)
        push!(fas,M)

        @test size(M) == (5, 4)
        @test mean(M) == mval
        @test P'P ≈ Matrix(I, 4, 4)
        @test all(isapprox.(cov(M), cov(X, dims=2), atol=1e-3))

        M = fit(FactorAnalysis, X; mean=mval, method=method)
        @test loadings(M) ≈ W

        M = fit(FactorAnalysis, Z; mean=0, method=method)
        @test loadings(M) ≈ W

        M = fit(FactorAnalysis, X; maxoutdim=3, method=method)
        P = projection(M)

        @test size(M) == (5, 3)
        @test P'P ≈ Matrix(I, 3, 3)
        end
    end

    # compare two algorithms
    M1, M2 = fas
    @test all(isapprox.(cov(M1), cov(M2), atol=1e-3)) # noise
    LL(m, x) = (-size(x,2)/2)*(size(x,1)*log(2π) + log(det(cov(m))) + tr(inv(cov(m))*cov(x, dims=2)))
    @test LL(M1, X) ≈ LL(M2, X) # log likelihood

    # test with different types
    XX = convert.(Float32, X)
    YY = convert.(Float32, Y)

    for method in (:cm, :em)
        MM = fit(FactorAnalysis, XX; method=method, maxoutdim=3)

        # mixing types
        predict(M, XX)
        predict(MM, X)
        reconstruct(M, YY)
        reconstruct(MM, Y)

        # type stability
        for func in (mean, projection, cov, var, loadings)
            @test eltype(func(M)) == Float64
            @test eltype(func(MM)) == Float32
        end
    end

    # views
    M = fit(FactorAnalysis, view(XX, :, 1:100), method=:cm, maxoutdim=3)
    M = fit(FactorAnalysis, view(XX, :, 1:100), method=:em, maxoutdim=3)
end
