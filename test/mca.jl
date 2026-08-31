using MultivariateStats, DataFrames, Test

function mca_simple_data()
    # Active variables
    da = DataFrame(
        :V1 => ["A", "A", "A", "A", "B", "B", "B", "B"],
        :V2 => ["X", "X", "Y", "X", "X", "Y", "X", "X"],
        :V3 => ["D", "D", "D", "C", "D", "C", "D", "C"],
    )

    # Passive variables
    dp = DataFrame(
        :V4 => [1, 2, 2, 2, 1, 1, 2, 1],
        :V5 => [2, 1, 2, 2, 1, 2, 1, 2],
    )

    return da, dp
end

# These values come from Stata and/or R FactoMineR.
function mca_simple_burt_correct()

    # Eigenvalues
    eig = [0.18724152, 0.11111111, 0.05473379]

    # Variable coordinates in standard normalization
    c1 = [1.0606602, -1.0606602, 0.35355339, -1.0606602, -1.5811388, 0.9486833]
    c2 = [0.8660254, -.8660254, -.8660254, 2.5980762, 0, 0]
    variable_standard = hcat(c1, c2)

    # Variable coordinates in principal normalization
    c1 = [0.45896265, -0.45896265, 0.15298755, -0.45896265, -0.68418112, 0.41050867]
    c2 = [0.28867513, -0.28867513, -0.28867513, 0.8660254, 0, 0]
    variable_principal = hcat(c1, c2)

    # Passive variable coordinates in standard normalization
    c1 = [-0.65213019, 0.65213019, 0.73080064, -0.43848039]
    c2 = [-0.4330127, 0.4330127, -1.1547005, 0.69282032]
    passive_standard = hcat(c1, c2)

    # Passive variable coordinates in principal normalization
    c1 = [-0.28218595, 0.28218595, 0.31622777, -0.18973666]
    c2 = [-0.14433757, 0.14433757, -0.38490018, 0.23094011]
    passive_principal = hcat(c1, c2)

    # Relative column contributions
    c1 = [0.187, 0.187, 0.031, 0.094, 0.312, 0.188]
    c2 = [0.125, 0.125, 0.188, 0.562, 0, 0]
    contrib = hcat(c1, c2)

    return (eig = eig,
            variable_principal = variable_principal,
            variable_standard = variable_standard,
            passive_principal = passive_principal,
            passive_standard = passive_standard,
            contrib = contrib)
end


# These values come from Stata and/or R FactoMiner.
function mca_simple_indicator_correct()

    # Eigenvalues in principal normalization
    eig = [0.4327141, 0.3333333, 0.2339525]

    # Object coordinates in principal normalization
    c1 = [
        7.876323e-01,
        7.876323e-01,
        0.3162278,
        -5.564176e-02,
        0.08052551,
        -1.2341531,
        0.08052551,
        -0.7627485,
    ]
    c2 = [0, 0, 1.1547005, 0, -0.57735027, 0.5773503, -0.57735027, -0.5773503]
    object_principal = hcat(c1, c2)

    # Variable coordinates in principal normalization
    c1 = [0.6977130, -0.6977130, 0.232571, -0.6977130, -1.040089e+00, 6.240535e-01]
    c2 = [0.5000000, -0.5000000, -0.500000, 1.5000000, 0, 0]
    variable_principal = hcat(c1, c2)

    # Variable coordinates in standard normalization
    c1 = [1.0606602, -1.0606602, 0.35355339, -1.0606602, -1.5811388, 0.9486833]
    c2 = [0.8660254, -0.8660254, -0.8660254, 2.5980762, 0, 0]
    variable_standard = hcat(c1, c2)

    # Passive variables in standard normalization
    c1 = [-0.65213019, 0.65213019, 0.73080064, -0.43848039]
    c2 = [-0.4330127, 0.4330127, -1.1547005, 0.69282032]
    passive_standard = hcat(c1, c2)

    # Passive variables in principal normalization
    c1 = [-0.42897783, 0.42897783, 0.48072805, -0.28843683]
    c2 = [-0.25, 0.25, -0.66666667, 0.4]
    passive_principal = hcat(c1, c2)

    # Squared column correlations
    c1 = [0.487, 0.487, 0.162, 0.162, 0.649, 0.649]
    c2 = [0.250, 0.250, 0.750, 0.750, 0, 0]
    sqcorr = hcat(c1, c2)

    # Relative column contributions
    c1 = [0.123, 0.123, 0.021, 0.062, 0.206, 0.123]
    c2 = [0.072, 0.072, 0.108, 0.325, 0, 0]
    contrib = hcat(c1, c2)

    return (eig = eig, object_principal = object_principal,
            variable_principal = variable_principal,
            variable_standard = variable_standard,
            passive_standard = passive_standard,
            passive_principal = passive_principal,
            sqcorr = sqcorr, contrib = contrib)
end

@testset "Compare MCA to R FactoMineR and Stata using Burt method" begin

    da, dp = mca_simple_data()
    xc = mca_simple_burt_correct()

    mp = fit(MCA, da; d=2, normalize = "principal", method = "burt")
    ms = fit(MCA, da; d=2, normalize = "standard", method = "burt")

    GPP = quali_passive(mp, dp; normalize = "principal")
    GPS = quali_passive(ms, dp; normalize = "standard")

    show(devnull, MIME("text/plain"), mp)
    show(devnull, MIME("text/plain"), ms)

    # Eigenvalues
    eig = inertia(mp).Raw[1:3]
    @test isapprox(eig, xc.eig, rtol = 1e-6, atol = 1e-6)

    # Variable coordinates in principal normalization
    GP = variable_coords(mp.C)
    @test isapprox(GP, xc.variable_principal, atol = 1e-6, rtol = 1e-6)

    # Variable coordinates in standard normalization
    GS = variable_coords(ms.C)
    @test isapprox(GS, xc.variable_standard, atol = 1e-6, rtol = 1e-6)

    # Passive variable coordinates in principal normalization
    #GPP = variable_coords(mp.C; type = "passive")
    @test isapprox(GPP.Coord, xc.passive_principal, atol = 1e-6, rtol = 1e-6)

    # Passive variable coordinates in standard normalization
    #GPS = variable_coords(ms.C; type = "passive")
    @test isapprox(GPS.Coord, xc.passive_standard, atol = 1e-6, rtol = 1e-6)

    for m in [mp, ms]
        stat = ca_stats(m.C)
        @test isapprox(stat.relcontrib_col, xc.contrib, atol = 1e-2, atol = 1e-2)
    end
end


@testset "Compare MCA to R FactoMineR and Stata using indicator method" begin

    da, dp = mca_simple_data()
    xc = mca_simple_indicator_correct()

    mp = fit(MCA, da; d=2, normalize = "principal", method = "indicator")
    ms = fit(MCA, da; d=2, normalize = "standard", method = "indicator")

    GPP = quali_passive(mp, dp, normalize = "principal")
    GPS = quali_passive(ms, dp, normalize = "standard")

    show(devnull, MIME("text/plain"), mp)
    show(devnull, MIME("text/plain"), ms)

    # Eigenvalues
    eig = inertia(mp).Raw[1:3]
    @test isapprox(eig, xc.eig, rtol = 1e-6, atol = 1e-6)

    # Variable coordinates in principal normalization
    G = variable_coords(mp.C; normalize = "principal")
    @test isapprox(G, xc.variable_principal, atol = 1e-6, rtol = 1e-6)

    # Variable coordinates in standard normalization
    GS = variable_coords(ms.C, normalize = "standard")
    @test isapprox(GS, xc.variable_standard, atol = 1e-6, rtol = 1e-6)

    # Object coordinates in principal normalization
    F = object_coords(mp; normalize = "principal")
    @test isapprox(F, xc.object_principal, atol = 1e-6, rtol = 1e-6)

    # Coordinates of passive variables in principal coordinates
    @test isapprox(GPP.Coord, xc.passive_principal, atol = 1e-6, rtol = 1e-6)

    # Coordinates of passive variables in standard coordinates
    @test isapprox(GPS.Coord, xc.passive_standard, atol = 1e-6, rtol = 1e-6)

    # Fit statistics
    for m in [mp, ms]
        stat = ca_stats(m.C)
        @test isapprox(stat.sqcorr_col, xc.sqcorr, atol = 1e-3, atol = 1e-3)
        @test isapprox(stat.relcontrib_col, xc.contrib, atol = 1e-2, atol = 1e-2)
    end
end
