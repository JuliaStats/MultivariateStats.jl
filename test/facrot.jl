using MultivariateStats
using LinearAlgebra
using Test
import Random

@testset "Factor Rotations" begin

    Random.seed!(37273)

    X = randn(5, 2)

    ## Varimax rotation
    # Comparison with R's `varimax` function called as
    # using RCall
    # R"vm <- varimax($X, normalize = FALSE, eps = 1e-12)"
    # R"F <- qm$loadings"
    # R"R <- qm$rotmat"
    # @rget F
    # @rget R

    F = [-2.56254778 -0.05712524;
         -0.37387139  0.53683094;
         -1.76868624  0.67513474;
          0.08057810  0.34826689;
         -0.06196338  0.20433233]
    R = [ 0.96256370  0.27105556;
         -0.27105556  0.96256370]

    FR = fit(FactorRotation, X)
    @test isapprox(FR.F, F, rtol = √eps())
    @test isapprox(FR.R, R, rtol = √eps())

    ## Quartimax rotation
    # Comparison with R's `GPArotation::quartimax` function called as
    # R"qm <- GPArotation::quartimax($X, eps = 1e-6, maxit = 10000)"
    # R"F <- qm$loadings"
    # R"R <- qm$Th"
    # @rget F
    # @rget R

    F = [-2.55665706 -0.18280898;
         -0.39976499  0.51783707;
         -1.79968636  0.58752613;
          0.06339043  0.35180152;
         -0.07191598  0.20104540]
    R = [ 0.94810241  0.31796513;
         -0.31796513  0.94810241]

    FR = fit(FactorRotation, X, alg = Orthomax(γ = 0.0))
    @test isapprox(FR.F, F, rtol = 1e-6)
    @test isapprox(FR.R, R, rtol = 1e-6)

    ## Equamax
    X = randn(10, 5)

    # Comparison with Matlab's `rotatefactors` called as
    # using MATLAB
    # mat"rotatefactors($X, 'Method', 'equamax')"

    F = [-0.01195028  0.05771308 -0.35271260  2.83724330 -0.68373015;
          0.07085630  2.43064454 -0.56171788  0.00250704  0.05441219;
         -0.12946985  0.78708715 -0.83039068 -2.01918172  0.65545648;
          1.95142102 -1.08251779 -0.49721317  1.13050103 -1.23799305;
         -0.55556677 -0.16288095  1.56941619 -0.36283530  2.46150446;
          0.21075154  2.16186107  1.16767716 -1.03674456  0.71948514;
          0.09579139  0.15765568  1.21512619  0.03140918  0.17671690;
          0.00855404  0.96679333 -1.69727390 -0.20320960 -0.75931306;
          0.01368675  0.34330631  0.00514439 -0.80566209  1.21579113;
         -2.13711263 -0.36494414 -0.25451404  0.25001421 -0.07785436]

    # Equamax for n x p matrix means γ = p / 2
    FR = fit(FactorRotation, X, alg = Orthomax(γ = 5 / 2, maxiter = 20000))
    @test isapprox(FR.F, F, rtol = √eps())

end