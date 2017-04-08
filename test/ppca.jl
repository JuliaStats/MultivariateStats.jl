using MultivariateStats
using Base.Test

srand(34568)

## PCA with zero mean

X = randn(5, 10)
Y = randn(3, 10)

W = qr(randn(5, 5))[1][:, 1:3]
σ² = 0.1
M = ProbPCA(Float64[], W, σ²)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == zeros(5)
@test projection(M) == W
@test var(M) == σ²

T = inv(W'*W .+ σ²*eye(3))*W'
@test_approx_eq transform(M, X[:,1]) T * X[:,1]
@test_approx_eq transform(M, X) T * X

R = W*inv(W'W)*(W'W .+ σ²*eye(3))
@test_approx_eq reconstruct(M, Y[:,1])  R * Y[:,1]
@test_approx_eq reconstruct(M, Y) R * Y


## PCA with non-zero mean

mv = rand(5)
M = ProbPCA(mv, W, σ²)

@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == mv
@test projection(M) == W
@test var(M) == σ²

@test_approx_eq transform(M, X[:,1]) T * (X[:,1] .- mv)
@test_approx_eq transform(M, X) T * (X .- mv)

@test_approx_eq reconstruct(M, Y[:,1])  R * Y[:,1] .+ mv
@test_approx_eq reconstruct(M, Y) R * Y .+ mv


## prepare training data

d = 5
n = 1000

R = qr(randn(d, d))[1]
@test_approx_eq R'R eye(5)
scale!(R, sqrt([0.5, 0.3, 0.1, 0.05, 0.05]))

X = R'randn(5, n) .+ randn(5)
mv = vec(mean(X, 2))
Z = X .- mv

M0 = fit(PCA, X; mean=mv, maxoutdim = 4)

## ppcaml (default)

M = fit(ProbPCA, X)
W = projection(M)

@test indim(M) == 5
@test outdim(M) == 4
@test mean(M) == mv
@test_approx_eq W'W diagm(diag(W'W))
@test_approx_eq reconstruct(M, transform(M, X)) reconstruct(M0, transform(M0, X))

M = fit(ProbPCA, X; mean=mv)
@test_approx_eq projection(M) W

M = fit(ProbPCA, Z; mean=0)
@test_approx_eq projection(M) W

M = fit(ProbPCA, X; maxoutdim=3)
W = projection(M)

@test indim(M) == 5
@test outdim(M) == 3
@test_approx_eq W'W diagm(diag(W'W))

# ppcaem

M = fit(ProbPCA, X; method=:em)
W = projection(M)
@test indim(M) == 5
@test outdim(M) == 4
@test mean(M) == mv
@test_approx_eq_eps reconstruct(M, transform(M, X)) reconstruct(M0, transform(M0, X)) 1e-3

M = fit(ProbPCA, X; method=:em, mean=mv)
@test_approx_eq projection(M) W

M = fit(ProbPCA, Z; method=:em, mean=0)
@test_approx_eq projection(M) W

M = fit(ProbPCA, X; method=:em, maxoutdim=3)
@test indim(M) == 5
@test outdim(M) == 3

# bayespca
M0 = fit(PCA, X; mean=mv, maxoutdim = 3)

M = fit(ProbPCA, X; method=:bayes)
W = projection(M)
@test indim(M) == 5
@test outdim(M) == 3
@test mean(M) == mv
@test_approx_eq reconstruct(M, transform(M, X)) reconstruct(M0, transform(M0, X))

M = fit(ProbPCA, X; method=:bayes, mean=mv)
@test_approx_eq projection(M) W

M = fit(ProbPCA, Z; method=:bayes, mean=0)
@test_approx_eq projection(M) W

M = fit(ProbPCA, X; method=:em, maxoutdim=2)
@test indim(M) == 5
@test outdim(M) == 2

# test that fit works with Float32 values
X2 = convert(Array{Float32,2}, X)
# Float32 input, default pratio
M = fit(ProbPCA, X2; maxoutdim=3)
