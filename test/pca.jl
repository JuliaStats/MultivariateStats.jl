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

@test transform(M, X[:,1]) ≈ P'X[:,1]
@test transform(M, @view X[:,1]) ≈ P'X[:,1]
@test transform(M, X) ≈ P'X

@test reconstruct(M, Y[:,1]) ≈ P * Y[:,1]
@test reconstruct(M, @view Y[:,1]) ≈ P * Y[:,1]
@test reconstruct(M, Y) ≈ P * Y


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

@test transform(M, X[:,1]) ≈ P' * (X[:,1] .- mv)
@test transform(M, X) ≈ P' * (X .- mv)

@test reconstruct(M, Y[:,1]) ≈ P * Y[:,1] .+ mv
@test reconstruct(M, Y) ≈ P * Y .+ mv


## prepare training data

d = 5
n = 1000

R = qr(randn(d, d))[1]
@test R'R ≈ eye(5)
scale!(R, sqrt.([0.5, 0.3, 0.1, 0.05, 0.05]))

X = R'randn(5, n) .+ randn(5)
mv = vec(mean(X, 2))
Z = X .- mv

C = cov(X, 2)
pvs0 = sort(eigvals(C); rev=true)
tv = sum(diag(C))

## pcacov (default using cov when d < n)

M = fit(PCA, X)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 5
@test outdim(M) == 5
@test mean(M) == mv
@test P'P ≈ eye(5)
@test C*P ≈ P*Diagonal(pvs)
@test issorted(pvs; rev=true)
@test pvs ≈ pvs0
@test tvar(M) ≈ tv
@test sum(pvs) ≈ tvar(M)
@test reconstruct(M, transform(M, X)) ≈ X

M = fit(PCA, X; mean=mv)
@test projection(M) ≈ P

M = fit(PCA, Z; mean=0)
@test projection(M) ≈ P

M = fit(PCA, X; maxoutdim=3)
P = projection(M)

@test indim(M) == 5
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; pratio=0.85)

@test indim(M) == 5
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

## pcastd

M = fit(PCA, X; method=:svd)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 5
@test outdim(M) == 5
@test mean(M) == mv
@test P'P ≈ eye(5)
@test isapprox(C*P, P*Diagonal(pvs), atol=1.0e-3)
@test issorted(pvs; rev=true)
@test isapprox(pvs, pvs0, atol=1.0e-3)
@test isapprox(tvar(M), tv, atol=1.0e-3)
@test sum(pvs) ≈ tvar(M)
@test reconstruct(M, transform(M, X)) ≈ X

M = fit(PCA, X; method=:svd, mean=mv)
@test projection(M) ≈ P

M = fit(PCA, Z; method=:svd, mean=0)
@test projection(M) ≈ P

M = fit(PCA, X; method=:svd, maxoutdim=3)
P = projection(M)

@test indim(M) == 5
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; method=:svd, pratio=0.85)

@test indim(M) == 5
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

# test that fit works with Float32 values
X2 = convert(Array{Float32,2}, X)
# Float32 input, default pratio
M = fit(PCA, X2; maxoutdim=3)
# Float32 input, specified Float64 pratio
M = fit(PCA, X2, pratio=0.85)
# Float32 input, specified Float32 pratio
M = fit(PCA, X2, pratio=0.85f0)
# Float64 input, specified Float32 pratio
M = fit(PCA, X, pratio=0.85f0)






## prepare training data
## with view
d = 5
n = 2000

R = qr(randn(d, d))[1]
@test R'R ≈ eye(5)
scale!(R, sqrt.([0.5, 0.3, 0.1, 0.05, 0.05]))

X = R'randn(5, n) .+ randn(5)
X = @view X[1:4, 1:1000]
mv = vec(mean(X, 2))
Z = X .- mv
mv = mv[1:4]

C = cov(X, 2)
pvs0 = sort(eigvals(C); rev=true)
tv = sum(diag(C))

## pcacov (default using cov when d < n)

M = fit(PCA, X)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 4
@test outdim(M) == 4
@test mean(M) == mv
@test P'P ≈ eye(4)
@test C*P ≈ P*Diagonal(pvs)
@test issorted(pvs; rev=true)
@test pvs ≈ pvs0
@test tvar(M) ≈ tv
@test sum(pvs) ≈ tvar(M)
@test reconstruct(M, transform(M, X)) ≈ X

M = fit(PCA, X; mean=mv)
@test projection(M) ≈ P

M = fit(PCA, Z; mean=0)
@test projection(M) ≈ P

M = fit(PCA, X; maxoutdim=3)
P = projection(M)

@test indim(M) == 4
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; pratio=0.85)

@test indim(M) == 4
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

## pcastd

M = fit(PCA, X; method=:svd)
P = projection(M)
pvs = principalvars(M)

@test indim(M) == 4
@test outdim(M) == 4
@test mean(M) == mv
@test P'P ≈ eye(4)
@test isapprox(C*P, P*Diagonal(pvs), atol=1.0e-3)
@test issorted(pvs; rev=true)
@test isapprox(pvs, pvs0, atol=1.0e-3)
@test isapprox(tvar(M), tv, atol=1.0e-3)
@test sum(pvs) ≈ tvar(M)
@test reconstruct(M, transform(M, X)) ≈ X

M = fit(PCA, X; method=:svd, mean=mv)
@test projection(M) ≈ P

M = fit(PCA, Z; method=:svd, mean=0)
@test projection(M) ≈ P

M = fit(PCA, X; method=:svd, maxoutdim=3)
P = projection(M)

@test indim(M) == 4
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

M = fit(PCA, X; method=:svd, pratio=0.85)

@test indim(M) == 4
@test outdim(M) == 3
@test P'P ≈ eye(3)
@test issorted(pvs; rev=true)

# test that fit works with Float32 values
X2 = convert(Array{Float32,2}, X)
# Float32 input, default pratio
M = fit(PCA, X2; maxoutdim=3)
# Float32 input, specified Float64 pratio
M = fit(PCA, X2, pratio=0.85)
# Float32 input, specified Float32 pratio
M = fit(PCA, X2, pratio=0.85f0)
# Float64 input, specified Float32 pratio
M = fit(PCA, X, pratio=0.85f0)
