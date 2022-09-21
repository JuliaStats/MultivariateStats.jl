using MultivariateStats, DataFrames, Test

@testset "Compare MCA to R FactoMiner" begin

    da = DataFrame(
        :V1 => ["A", "A", "A", "A", "B", "B", "B", "B"],
        :V2 => ["X", "X", "Y", "X", "X", "Y", "X", "X"],
        :V3 => ["D", "D", "D", "C", "D", "C", "D", "C"],
    )

    m = fit(MCA, da; d=3, vnames = names(da))
    F = objectscores(m)
    G = variablescores(m.C)

    # Eigenvalues
    eig = inertia(m).Raw[1:3]
    @test isapprox(eig, [0.4327141, 0.3333333, 0.2339525], rtol = 1e-6, atol = 1e-6)

    # Object coordinates
    c1 = [
        -7.876323e-01,
        -7.876323e-01,
        -0.3162278,
        5.564176e-02,
        -0.08052551,
        1.2341531,
        -0.08052551,
        0.7627485,
    ]
    c2 = [0, 0, 1.1547005, 0, -0.57735027, 0.5773503, -0.57735027, -0.5773503]
    c3 = [
        1.551768e-01,
        1.551768e-01,
        -0.3162278,
        9.984508e-01,
        -0.55193003,
        -0.1800605,
        -0.55193003,
        0.2913440,
    ]
    coord = hcat(c1, c2, c3)
    @test isapprox(coord, F, atol = 1e-6, rtol = 1e-6)

    # Variable coordinates
    c1 = [-0.6977130, 0.6977130, -0.232571, 0.6977130, 1.040089e+00, -6.240535e-01]
    c2 = [0.5000000, -0.5000000, -0.500000, 1.5000000, 0, 0]
    c3 = [0.5130269, -0.5130269, 0.171009, -0.5130269, 7.647753e-01, -4.588652e-01]
    coord = hcat(c1, c2, c3)
    @test isapprox(coord, G, atol = 1e-6, rtol = 1e-6)

end
