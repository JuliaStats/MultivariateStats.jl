using MultivariateStats
using Test
using LinearAlgebra
using StableRNGs

@testset "Regression" begin

    rng = StableRNG(34568)

    ## data

    m = 9
    n = 6
    n2 = 3

    X = randn(rng, m, n)
    A = randn(rng, n, n2)
    Xt = X'

    b = randn(rng, 1, n2)

    E = randn(rng, m, n2) * 0.1
    Y0 = X * A + E
    Y1 = X * A .+ b + E

    y0 = Y0[:,1]
    y1 = Y1[:,1]

    ## llsq

    A = llsq(X, Y0; dims=1, bias=false)
    A_r = copy(A)
    @test size(A) == (n, n2)
    @test X'Y0 ≈ X'X * A

    a = llsq(X, y0; dims=1, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    A = llsq(Xt, Y0; dims=2, bias=false)
    @test size(A) == (n, n2)
    @test A ≈ A_r

    a = llsq(Xt, y0; dims=2, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    Aa = llsq(X, Y1; dims=1, bias=true)
    Aa_r = copy(Aa)
    @test size(Aa) == (n+1, n2)
    A, b = Aa[1:end-1,:], Aa[end:end,:]
    @test X' * (Y1 .- b) ≈ X'X * A

    aa = llsq(X, y1; dims=1, bias=true)
    @test aa ≈ Aa[:,1]

    Aa = llsq(Xt, Y1; dims=2, bias=true)
    @test Aa ≈ Aa_r

    aa = llsq(Xt, y1; dims=2, bias=true)
    @test aa ≈ Aa[:,1]

    # test default dims=2 argument
    aa = llsq(X, y1)
    @test aa ≈ Aa[:,1]
    @test_throws DimensionMismatch llsq(X, y1; dims=2)
    @test_throws DimensionMismatch llsq(Xt, y1)


    ## ridge (with Real r)

    r = 0.1

    A = ridge(X, Y0, r; dims=1, bias=false)
    A_r = copy(A)
    @test size(A) == (n, n2)
    @test X'Y0 ≈ (X'X + r * I) * A

    a = ridge(X, y0, r; dims=1, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    A = ridge(Xt, Y0, r; dims=2, bias=false)
    @test size(A) == (n, n2)
    @test A ≈ A_r

    a = ridge(Xt, y0, r; dims=2, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    Aa = ridge(X, Y1, r; dims=1, bias=true)
    Aa_r = copy(Aa)
    @test size(Aa) == (n+1, n2)
    A, b = Aa[1:end-1,:], Aa[end:end,:]
    @test X' * (Y1 .- b) ≈ (X'X + r * I) * A

    aa = ridge(X, y1, r; dims=1, bias=true)
    @test aa ≈ Aa[:,1]

    Aa = ridge(Xt, Y1, r; dims=2, bias=true)
    @test Aa ≈ Aa_r

    aa = ridge(Xt, y1, r; dims=2, bias=true)
    @test aa ≈ Aa[:,1]

    # test default dims=2 argument
    aa = ridge(X, y1, r)
    @test aa ≈ Aa[:,1]
    @test_throws DimensionMismatch ridge(X, y1, r; dims=2)
    @test_throws DimensionMismatch ridge(Xt, y1, r)


    ## ridge (with diagonal r)

    r = 0.05 .+ 0.1 .* rand(rng, n)

    A = ridge(X, Y0, r; dims=1, bias=false)
    A_r = copy(A)
    @test size(A) == (n, n2)
    @test X'Y0 ≈ (X'X + diagm(0=>r)) * A

    a = ridge(X, y0, r; dims=1, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    A = ridge(Xt, Y0, r; dims=2, bias=false)
    @test size(A) == (n, n2)
    @test A ≈ A_r

    a = ridge(Xt, y0, r; dims=2, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    Aa = ridge(X, Y1, r; dims=1, bias=true)
    Aa_r = copy(Aa)
    @test size(Aa) == (n+1, n2)
    A, b = Aa[1:end-1,:], Aa[end:end,:]
    @test X' * (Y1 .- b) ≈ (X'X + diagm(0=>r)) * A

    aa = ridge(X, y1, r; dims=1, bias=true)
    @test aa ≈ Aa[:,1]

    Aa = ridge(Xt, Y1, r; dims=2, bias=true)
    @test Aa ≈ Aa_r

    aa = ridge(Xt, y1, r; dims=2, bias=true)
    @test aa ≈ Aa[:,1]


    ## ridge (with quadratic r matrix)

    Q = qr(randn(rng, n, n)).Q
    r = Q' * diagm(0=>r) * Q

    A = ridge(X, Y0, r; dims=1, bias=false)
    A_r = copy(A)
    @test size(A) == (n, n2)
    @test X'Y0 ≈ (X'X + r) * A

    a = ridge(X, y0, r; dims=1, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    A = ridge(Xt, Y0, r; dims=2, bias=false)
    @test size(A) == (n, n2)
    @test A ≈ A_r

    a = ridge(Xt, y0, r; dims=2, bias=false)
    @test size(a) == (n,)
    @test a ≈ A[:,1]

    Aa = ridge(X, Y1, r; dims=1, bias=true)
    Aa_r = copy(Aa)
    @test size(Aa) == (n+1, n2)
    A, b = Aa[1:end-1,:], Aa[end:end,:]
    @test X' * (Y1 .- b) ≈ (X'X + r) * A

    aa = ridge(X, y1, r; dims=1, bias=true)
    @test aa ≈ Aa[:,1]

    Aa = ridge(Xt, Y1, r; dims=2, bias=true)
    @test Aa ≈ Aa_r

    aa = ridge(Xt, y1, r; dims=2, bias=true)
    @test aa ≈ Aa[:,1]


    ## isotonic
    xx = [3.3, 3.3, 3.3, 6, 7.5, 7.5]
    yy = [4, 5, 1, 6, 8, 7.0]
    a = isotonic(xx, yy)
    b = [[1,2,3],[4],[5,6]]
    @test a ≈ xx atol=0.1

    ## using various types
    @testset for (T, TR) in [(Int,Float64), (Float32, Float32)]
        X, y = rand(T, 10, 3), rand(T, 10)
        a = llsq(X, y)
        @test eltype(a) == TR
        a = ridge(X, y, one(T))
        @test eltype(a) == TR
    end

end
