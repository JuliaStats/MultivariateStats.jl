using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
using Statistics: mean, cov, cor

@testset "CCA" begin

    rng = StableRNG(34568)

    dx = 5
    dy = 6
    p = 3

    # CCA with zero means

    X = rand(rng, dx, 100)
    Y = rand(rng, dy, 100)

    Px = qr(randn(rng, dx, dx)).Q[:, 1:p]
    Py = qr(randn(rng, dy, dy)).Q[:, 1:p]

    M = CCA(Float64[], Float64[], Px, Py, [0.8, 0.6, 0.4], zeros(3), -1)

    @test size(M)[1] == dx
    @test size(M)[2] == dy
    @test mean(M, :x) == zeros(dx)
    @test mean(M, :y) == zeros(dy)
    @test_throws ArgumentError mean(M, :z)
    @test projection(M, :x) == Px
    @test projection(M, :y) == Py
    @test_throws ArgumentError projection(M, :z)
    @test cor(M) == [0.8, 0.6, 0.4]

    @test predict(M, X, :x) ≈ Px'X
    @test predict(M, Y, :y) ≈ Py'Y

    ## CCA with nonzero means

    ux = randn(rng, dx)
    uy = randn(rng, dy)

    M = CCA(ux, uy, Px, Py, [0.8, 0.6, 0.4], zeros(3), -1)

    @test size(M)[1] == dx
    @test size(M)[2] == dy
    @test mean(M, :x) == ux
    @test mean(M, :y) == uy
    @test projection(M, :x) == Px
    @test projection(M, :y) == Py
    @test cor(M) == [0.8, 0.6, 0.4]

    @test predict(M, X, :x) ≈ Px' * (X .- ux)
    @test predict(M, Y, :y) ≈ Py' * (Y .- uy)


    ## prepare data

    n = 1000
    dg = 10
    G = randn(rng, dg, n)

    X = randn(rng, dx, dg) * G + 0.2 * randn(rng, dx, n)
    Y = randn(rng, dy, dg) * G + 0.2 * randn(rng, dy, n)
    xm = vec(mean(X, dims=2))
    ym = vec(mean(Y, dims=2))
    Zx = X .- xm
    Zy = Y .- ym

    Cxx = cov(X, dims=2)
    Cyy = cov(Y, dims=2)
    Cxy = cov(X, Y, dims=2)
    Cyx = Cxy'

    ## ccacov

    # X ~ Y
    M = fit(CCA, X, Y; method=:cov, outdim=p)
    Px = projection(M, :x)
    Py = projection(M, :y)
    rho = cor(M)
    @test size(M)[1] == dx
    @test size(M)[2] == dy
    @test size(M)[3] == p
    @test mean(M, :x) == xm
    @test mean(M, :y) == ym
    @test issorted(rho; rev=true)

    @test Px' * Cxx * Px ≈ Matrix(I, p, p)
    @test Py' * Cyy * Py ≈ Matrix(I, p, p)
    @test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
    @test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
    @test Px ≈ MultivariateStats.qnormalize!(Cxx \ (Cxy * Py), Cxx)
    @test Py ≈ MultivariateStats.qnormalize!(Cyy \ (Cyx * Px), Cyy)

    # Y ~ X
    M = fit(CCA, Y, X; method=:cov, outdim=p)
    Py = projection(M, :x)
    Px = projection(M, :y)
    rho = cor(M)
    @test size(M)[1] == dy
    @test size(M)[2] == dx
    @test size(M)[3] == p
    @test mean(M, :x) == ym
    @test mean(M, :y) == xm
    @test issorted(rho; rev=true)

    @test Px' * Cxx * Px ≈ Matrix(I, p, p)
    @test Py' * Cyy * Py ≈ Matrix(I, p, p)
    @test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
    @test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
    @test Px ≈ MultivariateStats.qnormalize!(Cxx \ (Cxy * Py), Cxx)
    @test Py ≈ MultivariateStats.qnormalize!(Cyy \ (Cyx * Px), Cyy)


    ## ccasvd

    # n > d
    M = fit(CCA, X, Y; method=:svd, outdim=p)
    Px = projection(M, :x)
    Py = projection(M, :y)
    rho = cor(M)
    @test size(M)[1] == dx
    @test size(M)[2] == dy
    @test size(M)[3] == p
    @test mean(M, :x) == xm
    @test mean(M, :y) == ym
    @test issorted(rho; rev=true)

    @test Px' * Cxx * Px ≈ Matrix(I, p, p)
    @test Py' * Cyy * Py ≈ Matrix(I, p, p)
    @test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
    @test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
    @test Px ≈ MultivariateStats.qnormalize!(Cxx \ (Cxy * Py), Cxx)
    @test Py ≈ MultivariateStats.qnormalize!(Cyy \ (Cyx * Px), Cyy)

    # different input types
    XX = convert.(Float32, X)
    YY = convert.(Float32, Y)

    MM = fit(CCA, view(XX, :, 1:400), view(YY, :, 1:400); method=:svd, outdim=p)

    # test that mixing types doesn't error
    predict(M, XX, :x)
    predict(M, YY, :y)
    predict(MM, XX, :x)
    predict(MM, YY, :y)

    # type stability
    for func in (M->mean(M, :x), M->mean(M, :y),
                 M->projection(M, :x),
                 M->projection(M, :y), cor)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end

    M1 = fit(CCA, X, Y; method=:svd, outdim=5)
    M2 = fit(CCA, X, Y; method=:cov, outdim=5)

    # From Stata
    stats = [0.000384245, 2.81275, 55.1432]
    df1 = [30, 30, 30]
    df2 = [3958, 4965, 4937]
    fstats = [810.3954, 212.8296, 1814.9480]

    ct1 = MultivariateStats.tests(M1)
    ct2 = MultivariateStats.tests(M2; n=size(X, 2))

    for ct in [ct1, ct2]
        @test isapprox(ct.stat, stats, atol=1e-5, rtol=1e-5)
        @test isapprox(ct.fstat, fstats, atol=1e-5, rtol=1e-5)
        @test isapprox(ct.df1, df1, atol=1e-5, rtol=1e-5)
        @test isapprox(ct.df2, df2, atol=1e-5, rtol=1e-5)
    end
end
