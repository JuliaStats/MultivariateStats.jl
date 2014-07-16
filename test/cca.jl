using MultivariateStats
using Base.Test

srand(34568)

dx = 6
dy = 5
p = 3

# CCA with zero means

X = rand(dx, 100)
Y = rand(dy, 100)

Px = qr(randn(dx, dx))[1][:, 1:p]
Py = qr(randn(dy, dy))[1][:, 1:p]

M = CCA(Float64[], Float64[], Px, Py, [0.8, 0.6, 0.4])

@test xindim(M) == dx
@test yindim(M) == dy
@test xmean(M) == zeros(dx)
@test ymean(M) == zeros(dy)
@test xprojection(M) == Px
@test yprojection(M) == Py
@test correlations(M) == [0.8, 0.6, 0.4]

@test_approx_eq xtransform(M, X) Px'X
@test_approx_eq ytransform(M, Y) Py'Y

## CCA with nonzero means

ux = randn(dx)
uy = randn(dy)

M = CCA(ux, uy, Px, Py, [0.8, 0.6, 0.4])

@test xindim(M) == dx
@test yindim(M) == dy
@test xmean(M) == ux
@test ymean(M) == uy
@test xprojection(M) == Px
@test yprojection(M) == Py
@test correlations(M) == [0.8, 0.6, 0.4]

@test_approx_eq xtransform(M, X) Px' * (X .- ux)
@test_approx_eq ytransform(M, Y) Py' * (Y .- uy)
