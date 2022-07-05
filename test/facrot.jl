using MultivariateStats
using Test
using StableRNGs

@testset "Factor Rotations" begin

    rng = StableRNG(37273)

    d = 5
    p = 2
    X = randn(rng, d, p)

    ## Varimax rotation
    # Ground-truth was extracted by specifying the following
    # print command
    # ````
    # using Printf
    # Base.show(io::IO, f::Float64) = @printf(io, "%1.8f", f)
    # ````
    # Comparison with R's `varimax` function called as
    # ````
    # using RCall
    # R"vm <- varimax($X, normalize = FALSE, eps = 1e-12)"
    # R"F <- vm$loadings"
    # R"R <- vm$rotmat"
    # @rget F
    # @rget R
    # ````

    F = [ 0.90503610  1.24332783;
         -2.32641022  0.36067316;
          0.97744296  1.36280123;
          0.85650467  0.23818243;
          0.45865489 -2.40335769]
    R = [ 0.89940646  0.43711327;
         -0.43711327  0.89940646]

    FR = rotate(X, Varimax())
    @test isapprox(FR[1], F, rtol = 1e-6)
    @test isapprox(FR[2], R, rtol = 1e-6)

    # Different equivalent ways of computing varimax
    FR = rotate(X, CrawfordFerguson{Orthogonal}(κ = 1.0 / 5.0))
    @test isapprox(FR[1], F, rtol = 1e-6)
    @test isapprox(FR[2], F, rtol = 1e-6)

    FR = rotate(X, Oblimin{Orthogonal}(γ = 1.0))
    @test isapprox(FR[1], F, rtol = 1e-6)
    @test isapprox(FR[2], F, rtol = 1e-6)

    ## Quartimax rotation
    # Comparison with R's `GPArotation::quartimax` function called as
    # ```
    # using RCall
    # R"qm <- GPArotation::quartimax($X, eps = 1e-6, maxit = 10000)"
    # R"F <- qm$loadings"
    # R"R <- qm$Th"
    # @rget F
    # @rget R
    # ```

    F = [ 0.89959033  1.24727370;
         -2.32796522  0.35049625;
          0.97147404  1.36706259;
          0.85545490  0.24192567;
          0.46916046 -2.40132899]
    R = [ 0.89748635  0.44104222;
         -0.44104222  0.89748635]

    FR = rotate(X, Quartimax())
    @test isapprox(loadings(FR), F, rtol = √eps(Float64))
    @test isapprox(rotation(FR), R, rtol = √eps(Float64))

    ## Equamax
    # Comparison with Matlab's `rotatefactors` called as
    # using MATLAB
    # F = mat"rotatefactors($X, 'Method', 'equamax', 'Normalize', 'off')"

    F = [ 0.90503611  1.24332782;
         -2.32641022  0.36067319;
          0.97744298  1.36280122;
          0.85650467  0.23818242;
          0.45865486 -2.40335769]

    # Equamax for d x p matrix means orthogonal Crawford-Ferguson with κ = p / (2 * d)
    FR = rotate(X, CrawfordFerguson{Orthogonal}(κ = p / (2 * d)))
    @test isapprox(loadings(FR), F, rtol = √eps(Float64))


    ## Parsimax

    # Comparison with Matlab's `rotatefactors` called as
    # ```
    # using MATLAB
    # F = mat"rotatefactors($X, 'Method', 'parsimax', 'Normalize', 'off')"
    # ```
    
    F = [ 0.90503611  1.24332782;
         -2.32641022  0.36067319;
          0.97744298  1.36280122;
          0.85650467  0.23818242;
          0.45865486 -2.40335769]

    # Parsimax for d x p matrix means orthogonal Crawford-Ferguson with κ = (p - 1) / (d + p - 2)
    FR = rotate(X, CrawfordFerguson{Orthogonal}(κ = (p - 1) / (d + p - 2)))
    @test isapprox(loadings(FR), F, rtol = √eps(Float64))

    ## Quartimin rotation
    # Comparison with R's `GPArotation::quartimin` function called as
    # ```
    # using RCall
    # R"qm <- GPArotation::quartimin($X, eps = 1e-6, maxit = 10000)"
    # R"F <- qm$loadings"
    # R"R <- qm$Th"
    # @rget F
    # @rget R
    # ```

    F = [ 0.94295134  1.29282920
         -2.32274063  0.24011390
          1.01896033  1.41630052
          0.86570192  0.28326547
          0.39161552 -2.38397278]
    R = [ 0.87548617  0.41144611
         -0.48324317  0.91143409]

    FR = rotate(X, Quartimin())
    @test isapprox(loadings(FR), F, rtol = √eps(Float64))
    @test isapprox(rotation(FR), R, rtol = √eps(Float64))

    # Test application to factor analysis and PCA models

    X = randn(10, 5)

    M = fit(FactorAnalysis, X)
    loadings(M)
    rotate!(M, Varimax())
    loadings(M)

    M = fit(PCA, X)
    projection(M)
    rotate!(M, Varimax())
    projection(M)
end