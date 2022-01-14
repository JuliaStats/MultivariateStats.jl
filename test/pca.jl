using MultivariateStats
using LinearAlgebra
using Test
using StableRNGs
import Statistics: mean, var, cov
import SparseArrays

@testset "PCA" begin

    rng = StableRNG(34568)

    ## PCA with zero mean

    X = randn(rng, 5, 10)
    Y = randn(rng, 3, 10)

    P = qr(randn(rng, 5, 5)).Q[:, 1:3]
    pvars = [5., 4., 3.]
    l = [-0.236399  -0.706558  -1.21106
          1.27426   -0.047799  -0.212282
          0.990598   1.22718   -0.882179
         -1.51861    0.725771  -0.631665
          0.181386  -1.21069   -0.557706]
    M = PCA(Float64[], P, pvars, 15.0)

    @test size(M) == (5, 3)
    @test mean(M) == zeros(5)
    @test projection(M) == P
    @test principalvars(M) == pvars
    @test principalvar(M, 2) == pvars[2]
    @test var(M) == 15.0
    @test tprincipalvar(M) == 12.0
    @test tresidualvar(M) == 3.0
    @test principalratio(M) == 0.8
    @test isapprox(loadings(M), l, atol = 0.001)

    @test predict(M, X[:,1]) ≈ P'X[:,1]
    @test predict(M, X) ≈ P'X

    @test reconstruct(M, Y[:,1]) ≈ P * Y[:,1]
    @test reconstruct(M, Y) ≈ P * Y


    ## PCA with non-zero mean

    mval = rand(rng, 5)
    M = PCA(mval, P, pvars, 15.0)

    @test size(M) == (5,3)
    @test mean(M) == mval
    @test projection(M) == P
    @test principalvars(M) == pvars
    @test principalvar(M, 2) == pvars[2]
    @test var(M) == 15.0
    @test tprincipalvar(M) == 12.0
    @test tresidualvar(M) == 3.0
    @test principalratio(M) == 0.8

    @test predict(M, X[:,1]) ≈ P' * (X[:,1] .- mval)
    @test predict(M, X) ≈ P' * (X .- mval)

    @test reconstruct(M, Y[:,1]) ≈ P * Y[:,1] .+ mval
    @test reconstruct(M, Y) ≈ P * Y .+ mval


    ## prepare training data

    d = 5
    n = 1000

    R = collect(qr(randn(rng, d, d)).Q)
    @test R'R ≈ Matrix(I, 5, 5)
    rmul!(R, Diagonal(sqrt.([0.5, 0.3, 0.1, 0.05, 0.05])))

    X = R'randn(rng, 5, n) .+ randn(rng, 5)
    mval = vec(mean(X, dims=2))
    Z = X .- mval

    C = cov(X, dims=2)
    pvs0 = sort(eigvals(C); rev=true)
    tv = sum(diag(C))

    ## pcacov (default using cov when d < n)

    M = fit(PCA, X)
    P = projection(M)
    pvs = principalvars(M)

    @test size(M) == (5,5)
    @test mean(M) == mval
    @test P'P ≈ Matrix(I, 5, 5)
    @test C*P ≈ P*Diagonal(pvs)
    @test issorted(pvs; rev=true)
    @test pvs ≈ pvs0
    @test var(M) ≈ tv
    @test sum(pvs) ≈ var(M)
    @test reconstruct(M, predict(M, X)) ≈ X

    M = fit(PCA, X; mean=mval)
    @test projection(M) ≈ P

    M = fit(PCA, Z; mean=0)
    @test projection(M) ≈ P

    M = fit(PCA, X; maxoutdim=3)
    P = projection(M)

    @test size(M) == (5,3)
    @test P'P ≈ Matrix(I, 3, 3)
    @test issorted(pvs; rev=true)

    M = fit(PCA, X; pratio=0.85)

    @test size(M) == (5,3)
    @test P'P ≈ Matrix(I, 3, 3)
    @test issorted(pvs; rev=true)

    ## pcastd

    M = fit(PCA, X; method=:svd)
    P = projection(M)
    pvs = principalvars(M)

    @test size(M) == (5,5)
    @test mean(M) == mval
    @test P'P ≈ Matrix(I, 5, 5)
    @test isapprox(C*P, P*Diagonal(pvs), atol=1.0e-3)
    @test issorted(pvs; rev=true)
    @test isapprox(pvs, pvs0, atol=1.0e-3)
    @test isapprox(var(M), tv, atol=1.0e-2)
    @test sum(pvs) ≈ var(M)
    @test reconstruct(M, predict(M, X)) ≈ X

    M = fit(PCA, X; method=:svd, mean=mval)
    @test projection(M) ≈ P

    M = fit(PCA, Z; method=:svd, mean=0)
    @test projection(M) ≈ P

    M = fit(PCA, X; method=:svd, maxoutdim=3)
    P = projection(M)

    @test size(M) == (5,3)
    @test P'P ≈ Matrix(I, 3, 3)
    @test issorted(pvs; rev=true)

    M = fit(PCA, X; method=:svd, pratio=0.85)

    @test size(M) == (5,3)
    @test P'P ≈ Matrix(I, 3, 3)
    @test issorted(pvs; rev=true)

    # Different data types
    # --------------------

    XX = convert.(Float32, X)
    YY = convert.(Float32, Y)
    p = 0.085
    pp = convert(Float32, p)

    MM = fit(PCA, XX; maxoutdim=3)

    # mix types
    fit(PCA, X ; pratio=pp)
    fit(PCA, XX ; pratio=p)
    fit(PCA, XX ; pratio=pp)
    predict(M, XX)
    predict(MM, X)
    reconstruct(M, YY)
    reconstruct(MM, Y)

    # type consistency
    for func in (mean, projection, principalvars, tprincipalvar, tresidualvar, tvar, principalratio)
        @test eltype(func(M)) == Float64
        @test eltype(func(MM)) == Float32
    end

    # views
    M = fit(PCA, view(X, :, 1:500), pratio=0.85)

    # sparse
    @test_throws AssertionError fit(PCA, SparseArrays.sprandn(rng, 100d, n, 0.6))

end
