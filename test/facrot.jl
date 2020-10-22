using MultivariateStats
using Test
import Random

@testset "Factor Rotations" begin

    Random.seed!(37273)

    X = randn(5, 2)

    ## Varimax rotation
    # Comparison with R's `varimax` function called as
    # using Printf
    # Base.show(io::IO, f::Float64) = @printf(io, "%1.8f", f)
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

    FR = fit(FactorRotation, X, alg = Varimax())
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

    FR = fit(FactorRotation, X, alg = Quartimax())
    @test isapprox(FR.F, F, rtol = 1e-6)
    @test isapprox(FR.R, R, rtol = 1e-6)

    # ## Equamax
    # X = randn(10, 5)

    # # Comparison with Matlab's `rotatefactors` called as
    # # using MATLAB
    # # mat"rotatefactors($X, 'Method', 'equamax', 'Normalize', 'off')"

    # F = [ 0.20295895  2.07838653 -0.54122597 -0.12487534  0.98358184;
    #      -1.03815211  0.17239352 -1.27261240 -0.41485812  1.44036802;
    #      -1.32942778  0.75883602  0.53246754 -0.39153390  2.81316627;
    #       0.17943030  0.10200669 -0.02036786 -2.05727725  0.36179405;
    #      -0.57150075  0.83976222  0.13897091 -0.58537812  1.78231347;
    #      -0.24047953 -0.14609212 -0.10808767 -0.31099008  0.53277677;
    #       1.63020394 -0.61318538 -0.77879612 -0.44644772 -0.30331055;
    #       2.92004862  0.09873379 -0.22817883 -0.00487858 -1.05015088;
    #       0.36654224  0.21341754 -2.75662510 -0.02498979 -0.47774001;
    #      -0.28002665 -0.92746802 -0.32811579 -0.52183972  0.15056833]

    # # Equamax for n x p matrix means γ = p / 2
    # FR = fit(FactorRotation, X, alg = Equamax(size(X, 2)))
    # @test isapprox(FR.F, F, rtol = √eps())

    # ## Parsimax

    # # Comparison with Matlab's `rotatefactors` called as
    # # using MATLAB
    # # mat"rotatefactors($X, 'Method', 'parsimax', 'Normalize', 'off')"

    # F = [ 0.20088755  2.13833957 -0.51952937 -0.15258917  0.85486221;
    #      -1.04430793  0.26969187 -1.25045141 -0.47113722  1.42298177;
    #      -1.32771913  0.92728472  0.57264722 -0.48321345  2.74040511;
    #       0.17856831  0.12930037 -0.00621592 -2.06837501  0.28192375;
    #      -0.57130614  0.94788128  0.16710829 -0.64273775  1.70426783;
    #      -0.24126103 -0.11168423 -0.10096713 -0.33074538  0.53024057;
    #       1.62645119 -0.62482282 -0.79024454 -0.44334505 -0.27314675;
    #       2.91932555  0.03504886 -0.25094949  0.03018747 -1.05060055;
    #       0.35443644  0.20160255 -2.76135234 -0.02259016 -0.46464651;
    #      -0.28200304 -0.91296776 -0.32805047 -0.53293768  0.19126898]

    # # Equamax for n x p matrix means γ = p / 2
    # FR = fit(FactorRotation, X, alg = Parsimax(size(X)..., maxiter = 2000, ϵ = 1e-5))
    # @test isapprox(FR.F, F, rtol = √eps())

end