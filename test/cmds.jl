using MultivariateStats
using LinearAlgebra
import Statistics: mean
using Test

@testset "MDS" begin

    d = 3
    n = 10
    X0 = randn(d, n)
    G0 = X0'X0
    D0 = MultivariateStats.pairwise((x,y)->norm(x-y), eachcol(X0), symmetric=true)

    ## conversion between dmat and gram

    @assert issymmetric(D0)
    @assert issymmetric(G0)

    D = gram2dmat(G0)
    @test issymmetric(D)
    @test D ≈ D0

    G = dmat2gram(D0)
    @test issymmetric(G)
    @test gram2dmat(G) ≈ D0

    ## classical MDS
    @test_throws UndefKeywordError fit(MDS, X0, maxoutdim=3)   # disambiguate distance matrix from data matrix
    M = fit(MDS, X0, maxoutdim=3, distances=false)
    @test size(M) == (d,3)
    @test size(projection(M)) == (n,3)
    @test length(eigvals(M)) == 3
    @test stress(M) ≈ 0.0 atol = 1e-10

    X = predict(M)
    @test size(X) == (3,n)
    @test MultivariateStats.pairwise((x,y)->norm(x-y), eachcol(X0), symmetric=true) ≈ D0

    @test_throws DimensionMismatch predict(M, rand(d+1))
    y = predict(M, X0[:, 1])
    @test X[:, 1] ≈ y

    # use only distance matrix
    M = fit(MDS, D0, maxoutdim=3, distances=true)
    @test isnan(size(M)[1])
    @test size(M)[2] == 3
    @test stress(M) ≈ 0.0 atol = 1e-10

    X = predict(M)
    @test size(X) == (3,n)
    @test MultivariateStats.pairwise((x,y)->norm(x-y), eachcol(X0), symmetric=true) ≈ D0

    @test_throws AssertionError predict(M, X0[:, 1])
    @test_throws DimensionMismatch predict(M, rand(d+1); distances = true)
    d = MultivariateStats.pairwise((x,y)->norm(x-y), eachcol(X0), eachcol(X0[:,2])) |> vec
    y = predict(M, d, distances=true)
    @test X[:, 2] ≈ y

    #Test MDS embeddings in dimensions >= number of points
    M = fit(MDS, [0. 1.; 1. 0.], maxoutdim=2, distances=true)
    @test size(M)[2] == 2
    @test predict(M) == [-0.5 0.5; 0 0]

    M = fit(MDS, [0. 1.; 1. 0.], maxoutdim=3, distances=true)
    @test size(M)[2] == 3
    @test predict(M) == [-0.5 0.5; 0 0; 0 0]


    #10 - dmat2gram produces negative definite matrix
    let D = [1.0 0.5 0.3181818181818182 0.38095238095238093 0.6111111111111112 0.36363636363636365 0.3333333333333333 0.4444444444444444 0.45454545454545453 0.38095238095238093
            0.5 1.0 0.3333333333333333 0.4166666666666667 0.32 0.37037037037037035 0.35 0.38461538461538464 0.38461538461538464 0.4
            0.3181818181818182 0.3333333333333333 1.0 0.38095238095238093 0.125 0.22727272727272727 0.47058823529411764 0.35294117647058826 0.2608695652173913 0.21052631578947367
            0.38095238095238093 0.4166666666666667 0.38095238095238093 1.0 0.17391304347826086 0.2962962962962963 0.38095238095238093 0.45454545454545453 0.3076923076923077 0.5454545454545454
            0.6111111111111112 0.32 0.125 0.17391304347826086 1.0 0.5 0.15 0.391304347826087 0.38095238095238093 0.3333333333333333
            0.36363636363636365 0.37037037037037035 0.22727272727272727 0.2962962962962963 0.5 1.0 0.2857142857142857 0.44 0.4 0.3181818181818182
            0.3333333333333333 0.35 0.47058823529411764 0.38095238095238093 0.15 0.2857142857142857 1.0 0.42105263157894735 0.45 0.23529411764705882
            0.4444444444444444 0.38461538461538464 0.35294117647058826 0.45454545454545453 0.391304347826087 0.44 0.42105263157894735 1.0 0.5416666666666666 0.5555555555555556
            0.45454545454545453 0.38461538461538464 0.2608695652173913 0.3076923076923077 0.38095238095238093 0.4 0.45 0.5416666666666666 1.0 0.4
            0.38095238095238093 0.4 0.21052631578947367 0.5454545454545454 0.3333333333333333 0.3181818181818182 0.23529411764705882 0.5555555555555556 0.4 1.0],
        Xt = [-0.27529104101488666 0.006134513718202863 0.33298809606740326 0.2608994458893664 -0.46185275796909575 -0.23734315039370618 0.29972782027671513 0.03827901455843394 -0.04096713097883363 0.07742518984640051
            -0.08177061420820278 -0.0044504235228030225 -0.3271919093638943 0.28206254638779243 -0.0954706915166714 -0.07137742126520012 -0.30754933764853587 0.18582658369448027 -0.03715307349750036 0.45707434094053534]',
        M = predict(fit(MDS, D, maxoutdim=2, distances=true))'
        @test M ≈ Xt .* cis.(angle.(sum(conj.(Xt) .* M, dims=1)))
    end

    #10 - test degenerate problem
    @test predict(fit(MDS, zeros(10, 10), maxoutdim=3, distances=false)) == zeros(3, 10)

    # out-of-sample
    D = [0 1 2 1;
         1 0 1 2;
         2 1 0 1;
         1 2 1 0.0f32]

    M = fit(MDS, sqrt.(D), maxoutdim=2, distances=true)
    X = predict(M)
    @test D ≈ MultivariateStats.pairwise((x,y)->sum(abs2, x-y), eachcol(X), symmetric=true)
    @test eltype(X) == Float32

    a = Float32[0.5, 0.5, 0.5, 0.5]
    A = vcat(hcat(D, a), hcat(a', zeros(Float32, 1, 1)))
    M⁺ = fit(MDS, sqrt.(A), maxoutdim=2, distances=true)
    X⁺ = predict(M⁺)
    @test A ≈ MultivariateStats.pairwise((x,y)->sum(abs2, x-y), eachcol(X⁺), symmetric=true)

    y = predict(M, a, distances=true)
    Y = [X y]
    @test A ≈ MultivariateStats.pairwise((x,y)->sum(abs2, x-y), eachcol(Y), symmetric=true)
    @test eltype(Y) == Float32

    # different input types
    d = 3
    X = randn(Float64, d, 10)
    XX = convert.(Float32, X)

    y = randn(Float64, d)
    yy = convert.(Float32, y)

    M = fit(MDS, X, maxoutdim=3, distances=false)
    MM = fit(MDS, XX, maxoutdim=3, distances=false)

    # test that mixing types doesn't error
    predict(M, yy)
    predict(MM, y)

    # type stability
    for func in (projection, eigvals, stress)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end
end
