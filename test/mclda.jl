using MultivariateStats
using LinearAlgebra
using Test
using StatsBase
import Statistics: mean, cov
import Random

@testset "Multi-class LDA" begin

    #Test equivalence of eigenvectors/singular vectors taking into account possible
    #phase (sign) differences
    #This may end up in Base.Test; see JuliaLang/julia#10651
    function test_approx_eq_vecs(a::StridedVecOrMat{S},
        b::StridedVecOrMat{T}, error=nothing) where {S<:Real, T<:Real}
        n = size(a, 2)
        @test n==size(b, 2) && size(a, 1)==size(b, 1)
        error==nothing && (error=n^3*(eps(S)+eps(T)))
        for i=1:n
            ev1, ev2 = a[:, i], b[:, i]
            deviation = min(abs(norm(ev1-ev2)), abs(norm(ev1+ev2)))
            if !isnan(deviation)
                @test isapprox.(deviation, 0.0, atol=error)
            end
        end
    end

    Random.seed!(34568)

    ## prepare data
    d = 5
    ns = [10, 15, 20]
    nc = length(ns)
    n = sum(ns)
    Xs = Matrix{Float64}[]
    ys = Vector{Int}[]
    Ss = Matrix{Float64}[]
    cmeans = zeros(d, nc)

    for k = 1:nc
        R = qr(randn(d, d)).Q
        nk = ns[k]

        Xk = R * Diagonal(2 * rand(d) .+ 0.5) * randn(d, nk) .+ randn(d)
        yk = fill(k, nk)
        uk = vec(mean(Xk, dims=2))
        Zk = Xk .- uk
        Sk = Zk * Zk'

        push!(Xs, Xk)
        push!(ys, yk)
        push!(Ss, Sk)
        cmeans[:,k] .= uk
    end

    X = hcat(Xs...)
    y = vcat(ys...)
    mv = vec(mean(X, dims=2))
    Sw = zeros(d, d)
    for k = 1:nc
        Sw += Ss[k]
    end

    Sb = zeros(d, d)
    for k = 1:nc
        dv = cmeans[:,k] - mv
        Sb += ns[k] * (dv * dv')
    end

    Z = X .- mv
    Sa = Z * Z'
    @test Sw + Sb ≈ Sa

    @assert size(X) == (d, n)
    @assert size(y) == (n,)

    ## Stats

    S = multiclass_lda_stats(nc, X, y)

    @test S.dim == d
    @test S.nclasses == nc

    @test classweights(S) == ns
    @test classmeans(S) ≈ cmeans
    @test mean(S) ≈ mv

    @test withclass_scatter(S) ≈ Sw
    @test betweenclass_scatter(S) ≈ Sb

    covestimator = SimpleCovariance(corrected=true)
    Sce = multiclass_lda_stats(nc, X, y; covestimator_between=covestimator, covestimator_within=covestimator)

    Swcorr = Sw * sum(ns)/(sum(ns)-1)
    Sbcorr = Sb * nc/(nc-1)

    @test withclass_scatter(Sce) ≈ Swcorr
    @test betweenclass_scatter(Sce) ≈ Sbcorr

    ## Solve

    emax = maximum(eigvals(Sw))
    Sw_c = copy(Sw)
    Sb_c = copy(Sb)

    lambda = 1.0e-3
    Sw_r = Sw + (lambda * emax) * Matrix(I, d, d)

    P1 = mclda_solve(Sb, Sw, :gevd, nc-1, lambda)
    @test P1' * Sw_r * P1 ≈ Matrix(I, nc-1, nc-1)
    U = Sb * P1
    V = Sw_r * P1
    # test whether U is proportional to V,
    # which indicates P is the generalized eigenvectors
    @test U ≈ V*Diagonal(vec(mean(U./V, dims=1)))

    P2 = mclda_solve(Sb, Sw, :whiten, nc-1, lambda)
    @test P2' * Sw_r * P2 ≈ Matrix(I, nc-1, nc-1)

    test_approx_eq_vecs(P1, P2)


    ## MC-LDA

    for T in (Float32, Float64)
        M = fit(MulticlassLDA, nc, convert(Matrix{T}, X), y; method=:gevd, regcoef=convert(T, lambda))
        @test size(M) == (d, nc - 1)
        @test projection(M) ≈ P1
        @test M.pmeans ≈ M.proj'cmeans
        @test predict(M, X) ≈ M.proj'X

        M = fit(MulticlassLDA, nc, convert(Matrix{T}, X), y; method=:whiten, regcoef=convert(T, lambda))
        @test size(M) == (d, nc - 1)
        # @test projection(M) P2  # signs may change
        @test M.pmeans ≈ M.proj'cmeans
        @test predict(M, X) ≈ M.proj'X
    end


    ## High-dimensional LDA (subspace LDA)

    # Low-dimensional case (no singularities)

    centers = [zeros(5) [10.0;zeros(4)] [0.0;10.0;zeros(3)]]

    # Case 1: 3 groups of 500
    dX = randn(5,1500);
    for i = 0:500:1000
        dX[:,(1:500).+i] .= dX[:,(1:500).+i] .- mean(dX[:,(1:500).+i], dims=2)  # make the mean of each 0
    end
    dX1 = dX
    X1 = [dX[:,1:500].+centers[:,1] dX[:,501:1000].+centers[:,2] dX[:,1001:1500].+centers[:,3]]
    label1 = [fill(1,500); fill(2,500); fill(3,500)]
    # Case 2: 3 groups, one with 1000, one with 100, and one with 10
    dX = randn(5,1110);
    dX[:,   1:1000] .= dX[:,   1:1000] .- mean(dX[:,   1:1000], dims=2)
    dX[:,1001:1100] .= dX[:,1001:1100] .- mean(dX[:,1001:1100], dims=2)
    dX[:,1101:1110] .= dX[:,1101:1110] .- mean(dX[:,1101:1110], dims=2)
    dX2 = dX
    X2 = [dX[:,1:1000].+centers[:,1] dX[:,1001:1100].+centers[:,2] dX[:,1101:1110].+centers[:,3]]
    label2 = [fill(1,1000); fill(2,100); fill(3,10)]

    for (X, dX, label) in ((X1, dX1, label1), (X2, dX2, label2))
        w = [length(findall(label.==1)), length(findall(label.==2)), length(findall(label.==3))]
        M = fit(SubspaceLDA, X, label)
        @test indim(M) == 5
        @test outdim(M) == 2
        totcenter = vec(sum(centers.*w', dims=2)./sum(w))
        @test mean(M) ≈ totcenter
        @test classmeans(M) ≈ centers
        @test classweights(M) == w
        x = rand(5)
        @test predict(M, x) ≈ projection(M)'*x
        dcenters = centers .- totcenter
        Hb = dcenters.*sqrt.(w)'
        Sb = Hb*Hb'
        Sw = dX*dX'
        # When there are no singularities, we need to solve the "regular" LDA eigenvalue equation
        proj = projection(M)
        @test Sb*proj ≈ Sw*proj*Diagonal(M.λ)
        @test proj'*Sw*proj ≈ Matrix(I,2,2)

        # also check that this is consistent with the conventional algorithm
        Mld = fit(MulticlassLDA, 3, X, label)
        for i = 1:2
            @test abs(dot(normalize(proj[:,i]), normalize(Mld.proj[:,i]))) ≈ 1
        end
    end

    # High-dimensional case (undersampled => singularities)
    X = randn(10^6, 9)
    label = rand(1:3, 9); label[1:3] = 1:3
    M = fit(SubspaceLDA, X, label)
    centers = M.cmeans
    for i = 1:3
        flag = label.==i
        @test centers[:,i] ≈ mean(X[:,flag], dims=2)
        @test M.cweights[i] == sum(flag)
    end
    dcenters = centers .- mean(X, dims=2)
    Hb = dcenters*Diagonal(sqrt.(M.cweights))
    Hw = X - centers[:,label]
    @test (M.projw'*Hb)*(Hb'*M.projw)*M.projLDA ≈ (M.projw'*Hw)*(Hw'*M.projw)*M.projLDA*Diagonal(M.λ)
    Gw = projection(M)'*Hw
    @test Gw*Gw' ≈ Matrix(I,2,2)

    # Test that nothing breaks if one class has no members
    label[1:4] .= 1
    label[5:9] .= 3  # no data points have label == 2
    M = fit(SubspaceLDA, X, label)
    centers = M.cmeans
    dcenters = centers .- mean(X, dims=2)
    Hb = dcenters*Diagonal(sqrt.(M.cweights))
    Hw = X - centers[:,label]
    @test (M.projw'*Hb)*(Hb'*M.projw)*M.projLDA ≈ (M.projw'*Hw)*(Hw'*M.projw)*M.projLDA*Diagonal(M.λ)

    # Test normalized LDA
    function gen_ldadata_2(centers, n1, n2)
        d = size(centers, 1)
        X = randn(d, n1+n2)
        X[:,1:n1]       .-= vec(mean(X[:,1:n1], dims=2))
        X[:,n1+1:n1+n2] .-= vec(mean(X[:,n1+1:n1+n2], dims=2))
        dX = copy(X)
        X[:,1:n1]       .+= centers[:,1]
        X[:,n1+1:n1+n2] .+= centers[:,2]
        label = [fill(1,n1); fill(2,n2)]
        X, dX, label
    end
    centers = zeros(2,2); centers[1,2] = 10
    n1 = 100
    for n2 in (100, 1000, 10000)
        X, dX, label = gen_ldadata_2(centers, n1, n2)
        M = fit(SubspaceLDA, X, label; normalize=true)
        proj = projection(M)
        Hb = centers .- mean(centers, dims=2)   # not weighted by number in each cluster
        Sb = Hb*Hb'
        Sw = dX*dX'/(n1+n2)
        @test Sb*proj ≈ Sw*proj*Diagonal(M.λ)
    end

    # Test various input data types
    for (T, nrm) in Iterators.product((Float64, Float32), (false, true))
        n2 = 100
        X, dX, label = gen_ldadata_2(centers, n1, n2)
        M = fit(SubspaceLDA, convert(Matrix{T}, X), label; normalize=nrm)
        proj = projection(M)
        @test eltype(proj) === T
    end

end
