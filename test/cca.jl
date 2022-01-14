using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
import Statistics: mean, cov


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

    M = CCA(Float64[], Float64[], Px, Py, [0.8, 0.6, 0.4])

    @test xindim(M) == dx
    @test yindim(M) == dy
    @test xmean(M) == zeros(dx)
    @test ymean(M) == zeros(dy)
    @test xprojection(M) == Px
    @test yprojection(M) == Py
    @test correlations(M) == [0.8, 0.6, 0.4]

    @test xtransform(M, X) ≈ Px'X
    @test ytransform(M, Y) ≈ Py'Y

    ## CCA with nonzero means

    ux = randn(rng, dx)
    uy = randn(rng, dy)

    M = CCA(ux, uy, Px, Py, [0.8, 0.6, 0.4])

    @test xindim(M) == dx
    @test yindim(M) == dy
    @test xmean(M) == ux
    @test ymean(M) == uy
    @test xprojection(M) == Px
    @test yprojection(M) == Py
    @test correlations(M) == [0.8, 0.6, 0.4]

    @test xtransform(M, X) ≈ Px' * (X .- ux)
    @test ytransform(M, Y) ≈ Py' * (Y .- uy)


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
    Px = xprojection(M)
    Py = yprojection(M)
    rho = correlations(M)
    @test xindim(M) == dx
    @test yindim(M) == dy
    @test outdim(M) == p
    @test xmean(M) == xm
    @test ymean(M) == ym
    @test issorted(rho; rev=true)

    @test Px' * Cxx * Px ≈ Matrix(I, p, p)
    @test Py' * Cyy * Py ≈ Matrix(I, p, p)
    @test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
    @test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
    @test Px ≈ MultivariateStats.qnormalize!(Cxx \ (Cxy * Py), Cxx)
    @test Py ≈ MultivariateStats.qnormalize!(Cyy \ (Cyx * Px), Cyy)

    # Y ~ X
    M = fit(CCA, Y, X; method=:cov, outdim=p)
    Py = xprojection(M)
    Px = yprojection(M)
    rho = correlations(M)
    @test xindim(M) == dy
    @test yindim(M) == dx
    @test outdim(M) == p
    @test xmean(M) == ym
    @test ymean(M) == xm
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
    Px = xprojection(M)
    Py = yprojection(M)
    rho = correlations(M)
    @test xindim(M) == dx
    @test yindim(M) == dy
    @test outdim(M) == p
    @test xmean(M) == xm
    @test ymean(M) == ym
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
    xtransform(M, XX)
    ytransform(M, YY)
    xtransform(MM, XX)
    ytransform(MM, YY)
    
    # type stability
    for func in (xmean, ymean, xprojection, yprojection, correlations)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end
end
