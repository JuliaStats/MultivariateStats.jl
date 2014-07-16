using MultivariateStats
using Base.Test

srand(34568)

## PCA with zero mean

X = randn(5, 10)
Y = randn(3, 10)

P = qr(randn(5, 5))[1][:, 1:3]
pvars = [5., 4., 3.]
M = PCA(Float64[], P, pvars, 15.0)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == zeros(5)
@test projection(M) == P
@test principalvars(M) == pvars
@test principalvar(M, 2) == pvars[2]
@test tvar(M) == 15.0
@test tprincipalvar(M) == 12.0
@test tresidualvar(M) == 3.0
@test principalratio(M) == 0.8

@test_approx_eq transform(M, X[:,1]) P'X[:,1]
@test_approx_eq transform(M, X) P'X

@test_approx_eq reconstruct(M, Y[:,1]) P * Y[:,1]
@test_approx_eq reconstruct(M, Y) P * Y


## PCA with non-zero mean

mv = rand(5)
M = PCA(mv, P, pvars, 15.0)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == mv
@test projection(M) == P
@test principalvars(M) == pvars
@test principalvar(M, 2) == pvars[2]
@test tvar(M) == 15.0
@test tprincipalvar(M) == 12.0
@test tresidualvar(M) == 3.0
@test principalratio(M) == 0.8

@test_approx_eq transform(M, X[:,1]) P' * (X[:,1] .- mv)
@test_approx_eq transform(M, X) P' * (X .- mv)

@test_approx_eq reconstruct(M, Y[:,1]) P * Y[:,1] .+ mv
@test_approx_eq reconstruct(M, Y) P * Y .+ mv


## prepare training data

d = 5
n = 1000

R = qr(randn(d, d))[1]
@test_approx_eq R'R eye(5)
scale!(R, sqrt([0.5, 0.3, 0.1, 0.05, 0.05]))

X = R'randn(5, n) .+ randn(5)
mv = vec(mean(X, 2))
Z = X .- mv

C = cov(X; vardim=2)
pvs0 = sort(eigvals(C); rev=true)
tv = sum(diag(C))

## pcacov (default using cov when d < n)

M = fit(PCA, X)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 5
@test outdim(M) == 5
@test mean(M) == mv
@test_approx_eq P'P eye(5)
@test issorted(pvs; rev=true)
@test_approx_eq pvs pvs0
@test_approx_eq tvar(M) tv
@test_approx_eq sum(pvs) tvar(M)
@test_approx_eq reconstruct(M, transform(M, X)) X

M = fit(PCA, X; mean=mv)
@test_approx_eq projection(M) P

M = fit(PCA, Z; mean=0)
@test_approx_eq projection(M) P

M = fit(PCA, X; maxoutdim=3)
P = projection(M)

@test indim(M) == 5
@test outdim(M) == 3
@test_approx_eq P'P eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; pratio=0.85)

@test indim(M) == 5
@test outdim(M) == 3
@test_approx_eq P'P eye(3)
@test issorted(pvs; rev=true)

## pcastd

M = fit(PCA, X; method=:std)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 5
@test outdim(M) == 5
@test mean(M) == mv
@test_approx_eq P'P eye(5)
@test issorted(pvs; rev=true)
@test_approx_eq_eps pvs pvs0 1.0e-3
@test_approx_eq_eps tvar(M) tv 1.0e-3
@test_approx_eq sum(pvs) tvar(M)
@test_approx_eq reconstruct(M, transform(M, X)) X

M = fit(PCA, X; method=:std, mean=mv)
@test_approx_eq projection(M) P

M = fit(PCA, Z; method=:std, mean=0)
@test_approx_eq projection(M) P

M = fit(PCA, X; method=:std, maxoutdim=3)
P = projection(M)

@test indim(M) == 5
@test outdim(M) == 3
@test_approx_eq P'P eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; method=:std, pratio=0.85)

@test indim(M) == 5
@test outdim(M) == 3
@test_approx_eq P'P eye(3)
@test issorted(pvs; rev=true)

