using MultivariateStats
using Base.Test

srand(34568)

## data

d = 3
n = 5

X = rand(d, n)
mv = vec(mean(X, 2))
C = cov(X, 2)
C0 = copy(C)

emax = maximum(eigvals(C))
rc = 0.01
Cr = C + (emax * rc) * eye(d)

## cov_whitening

cf = cholfact(C, :U)
W = cov_whitening(cf)
@test istriu(W)
@test W'C * W ≈ eye(d)

cf = cholfact(C, :L)
W = cov_whitening(cf)
@test istriu(W)
@test W'C * W ≈ eye(d)

W = cov_whitening(C)
@test C == C0
@test istriu(W)
@test W'C * W ≈ eye(d)

W = cov_whitening(C, 0.0)
@test C == C0
@test istriu(W)
@test W'C * W ≈ eye(d)

W = cov_whitening(C, rc)
@test istriu(W)
@test W'Cr * W ≈ eye(d)


# Whitening from data

f = fit(Whitening, X)
W = f.W
@test isa(f, Whitening{Float64})
@test mean(f) === f.mean
@test istriu(W)
@test W'C * W ≈ eye(d)
@test transform(f, X) ≈ W' * (X .- f.mean)

f = fit(Whitening, X; regcoef=rc)
W = f.W
@test W'Cr * W ≈ eye(d)

f = fit(Whitening, X; mean=mv)
W = f.W
@test W'C * W ≈ eye(d)

f = fit(Whitening, X; mean=0)
Cx = (X * X') / (n - 1)
W = f.W
@test W'Cx * W ≈ eye(d)

# invsqrtm

R = invsqrtm(C)
@test C == C0
@test R ≈ inv(sqrtm(C))

