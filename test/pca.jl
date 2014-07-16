using MultivariateStats
using Base.Test

srand(34568)

## prepare data

# d = 5
# n = 1000

# R = qr(randn(d, d))[1]
# @test_approx_eq R'R eye(5)
# scale!(R, rand(d) .+ 0.5)

# center = randn(5)

# Z = R'randn(5, n)
# X = Z .+ center
# w = rand(n)

# Xcopy = copy(X)  


## PCA with zero mean

X = randn(5, 10)
Y = randn(3, 10)

P = qr(randn(5, 5))[1][:, 1:3]
pvars = [5., 4., 3.]
M = PCA(Float64[], P, pvars, 0.25)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == zeros(5)
@test principalvars(M) == pvars
@test principalvar(M, 2) == pvars[2]
@test tprincipalvar(M) == 12.0
@test tresidualvar(M) == 0.25
@test principalratio(M) == 12.0 / 12.25

@test_approx_eq transform(M, X[:,1]) P'X[:,1]
@test_approx_eq transform(M, X) P'X

@test_approx_eq reconstruct(M, Y[:,1]) P * Y[:,1]
@test_approx_eq reconstruct(M, Y) P * Y


## PCA with non-zero mean

mv = rand(5)
M = PCA(mv, P, pvars, 0.25)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == mv
@test principalvars(M) == pvars
@test principalvar(M, 2) == pvars[2]
@test tprincipalvar(M) == 12.0
@test tresidualvar(M) == 0.25
@test principalratio(M) == 12.0 / 12.25

@test_approx_eq transform(M, X[:,1]) P' * (X[:,1] .- mv)
@test_approx_eq transform(M, X) P' * (X .- mv)

@test_approx_eq reconstruct(M, Y[:,1]) P * Y[:,1] .+ mv
@test_approx_eq reconstruct(M, Y) P * Y .+ mv






