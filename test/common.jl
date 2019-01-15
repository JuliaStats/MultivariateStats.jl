@testset "data transforms" begin
    X = rand(5, 100)
    m = vec(mean(X, dims = 2))
    s = vec(std(X, dims = 2))

    Xz = (X .- m) ./ s
    Xm = X .- m
    Xs = X ./ s

    @test MultivariateStats.centralize(X, m) ≈ Xm
    @test MultivariateStats.standardize(X, s) ≈ Xs
    @test MultivariateStats.centralize(X, []) ≈ X
    @test MultivariateStats.standardize(X, []) ≈ X
    @test MultivariateStats.ztransform(X, m, s) ≈ Xz
    @test MultivariateStats.ztransform(X, [], []) ≈ X
    @test MultivariateStats.ztransform(X, m, []) ≈ Xm
    @test MultivariateStats.ztransform(X, [], s) ≈ Xs

    @test MultivariateStats.decentralize(Xm, m) ≈ X
    @test MultivariateStats.destandardize(Xs, s) ≈ X
    @test MultivariateStats.decentralize(Xm, []) ≈ Xm
    @test MultivariateStats.destandardize(Xs, []) ≈ Xs
    @test MultivariateStats.deztransform(Xz, m, s) ≈ X
    @test MultivariateStats.deztransform(X, [], []) ≈ X
    @test MultivariateStats.deztransform(Xm, m, []) ≈ X
    @test MultivariateStats.deztransform(Xs, [], s) ≈ X

    @test MultivariateStats.centralize(X[:, 1], m) ≈ Xm[:, 1]
    @test MultivariateStats.standardize(X[:, 1], s) ≈ Xs[:, 1]
    @test MultivariateStats.centralize(X[:, 1], []) ≈ X[:, 1]
    @test MultivariateStats.standardize(X[:, 1], []) ≈ X[:, 1]
    @test MultivariateStats.ztransform(X[:, 1], m, s) ≈ Xz[:, 1]
    @test MultivariateStats.ztransform(X[:, 1], [], []) ≈ X[:, 1]
    @test MultivariateStats.ztransform(X[:, 1], m, []) ≈ Xm[:, 1]
    @test MultivariateStats.ztransform(X[:, 1], [], s) ≈ Xs[:, 1]

    @test MultivariateStats.decentralize(Xm[:, 1], m) ≈ X[:, 1]
    @test MultivariateStats.destandardize(Xs[:, 1], s) ≈ X[:, 1]
    @test MultivariateStats.decentralize(Xm[:, 1], []) ≈ Xm[:, 1]
    @test MultivariateStats.destandardize(Xs[:, 1], []) ≈ Xs[:, 1]
    @test MultivariateStats.deztransform(Xz[:, 1], m, s) ≈ X[:, 1]
    @test MultivariateStats.deztransform(X[:, 1], [], []) ≈ X[:, 1]
    @test MultivariateStats.deztransform(Xm[:, 1], m, []) ≈ X[:, 1]
    @test MultivariateStats.deztransform(Xs[:, 1], [], s) ≈ X[:, 1]
end
