using MultivariateStats
using Base.Test

import MultivariateStats: qnormalize!

srand(34568)

dx = 5
dy = 6
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

@test xtransform(M, X) ≈ Px'X
@test ytransform(M, Y) ≈ Py'Y

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

@test xtransform(M, X) ≈ Px' * (X .- ux)
@test ytransform(M, Y) ≈ Py' * (Y .- uy)


## prepare data

n = 1000
dg = 10
G = randn(dg, n)

X = randn(dx, dg) * G + 0.2 * randn(dx, n)
Y = randn(dy, dg) * G + 0.2 * randn(dy, n)
xm = vec(mean(X, 2))
ym = vec(mean(Y, 2))
Zx = X .- xm
Zy = Y .- ym

Cxx = cov(X, 2)
Cyy = cov(Y, 2)
Cxy = cov(X, Y, 2)
Cyx = Cxy'

## ccacov

# X ~ Y
M = fit(CCA, X, Y; method=:cov, outdim=p)
Px = xprojection(M)
Py = yprojection(M)
rho = correlations(M)
@test xindim(M) == dx
@test yindim(M) == dy
@test outdim(M) == p
@test xmean(M) == xm
@test ymean(M) == ym
@test issorted(rho; rev=true)

@test Px' * Cxx * Px ≈ eye(p)
@test Py' * Cyy * Py ≈ eye(p)
@test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
@test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
@test Px ≈ qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test Py ≈ qnormalize!(Cyy \ (Cyx * Px), Cyy)

# Y ~ X
M = fit(CCA, Y, X; method=:cov, outdim=p)
Py = xprojection(M)
Px = yprojection(M)
rho = correlations(M)
@test xindim(M) == dy
@test yindim(M) == dx
@test outdim(M) == p
@test xmean(M) == ym
@test ymean(M) == xm
@test issorted(rho; rev=true)

@test Px' * Cxx * Px ≈ eye(p)
@test Py' * Cyy * Py ≈ eye(p)
@test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
@test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
@test Px ≈ qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test Py ≈ qnormalize!(Cyy \ (Cyx * Px), Cyy)


## ccasvd

# n > d
M = fit(CCA, X, Y; method=:svd, outdim=p)
Px = xprojection(M)
Py = yprojection(M)
rho = correlations(M)
@test xindim(M) == dx
@test yindim(M) == dy
@test outdim(M) == p
@test xmean(M) == xm
@test ymean(M) == ym
@test issorted(rho; rev=true)

@test Px' * Cxx * Px ≈ eye(p)
@test Py' * Cyy * Py ≈ eye(p)
@test Cxy * (Cyy \ Cyx) * Px ≈ Cxx * Px * Diagonal(rho.^2)
@test Cyx * (Cxx \ Cxy) * Py ≈ Cyy * Py * Diagonal(rho.^2)
@test Px ≈ qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test Py ≈ qnormalize!(Cyy \ (Cyx * Px), Cyy)
