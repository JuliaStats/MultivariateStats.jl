using MultivariateStats
using Test
using StableRNGs

@testset "Factor Rotations" begin

    rng = StableRNG(37273)

    X = randn(rng, 5, 2)

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

    FR = fit(FactorRotation, X, alg = Varimax())
    @test isapprox(FR.F, F, rtol = √eps(Float64))
    @test isapprox(FR.R, R, rtol = √eps(Float64))

    isapprox(gpaortho(X, varimax, 1000, 10, 1e-6)[1], F, rtol = 1e-6)
    isapprox(gpaortho(X, varimax, 1000, 10, 1e-6)[2], R, rtol = 1e-6)

    FR = rotate(X, CrawfordFerguson{Orthogonal}(κ = 1.0 / 5.0))
    isapprox(FR[1], F, rtol = 1e-6)
    isapprox(FR[2], F, rtol = 1e-6)

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

    FR = fit(FactorRotation, X, alg = Quartimax())
    @test isapprox(FR.F, F, rtol = √eps(Float64))
    @test isapprox(FR.R, R, rtol = √eps(Float64))

    isapprox(gpaortho(X, quartimax, 1000, 10, 1e-6)[1], F, rtol = √eps(Float64))
    isapprox(gpaortho(X, quartimax, 1000, 10, 1e-6)[2], R, rtol = √eps(Float64))

    ## Equamax
    # Comparison with Matlab's `rotatefactors` called as
    # using MATLAB
    # F = mat"rotatefactors($X, 'Method', 'equamax', 'Normalize', 'off')"

    F = [ 0.90503611  1.24332782;
         -2.32641022  0.36067319;
          0.97744298  1.36280122;
          0.85650467  0.23818242;
          0.45865486 -2.40335769]

    # Equamax for n x p matrix means γ = p / 2
    FR = fit(FactorRotation, X, alg = Equamax(size(X, 2)))
    @test isapprox(FR.F, F, rtol = √eps(Float64))


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

    # Equamax for n x p matrix means γ = p / 2
    FR = fit(FactorRotation, X, alg = Parsimax(size(X)...))
    @test isapprox(FR.F, F, rtol = √eps(Float64))

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

    isapprox(gpaoblique(X, quartimin, 1000, 50, 1e-6)[1], F, rtol = √eps(Float64))
    isapprox(gpaoblique(X, quartimin, 1000, 10, 1e-6)[2], R, rtol = √eps(Float64))

end