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

Cxx = cov(X; vardim=2)
Cyy = cov(Y; vardim=2)
Cxy = cov(X, Y; vardim=2)
Cyx = Cxy'

## ccacov

# X ~ Y
M = ccacov(Cxx, Cyy, Cxy, xm, ym, p)
Px = xprojection(M)
Py = yprojection(M)
rho = correlations(M)
@test xindim(M) == dx
@test yindim(M) == dy
@test outdim(M) == p
@test xmean(M) == xm
@test ymean(M) == ym
@test issorted(rho; rev=true)

@test_approx_eq Px' * Cxx * Px eye(p)
@test_approx_eq Py' * Cyy * Py eye(p)
@test_approx_eq Cxy * (Cyy \ Cyx) * Px scale(Cxx * Px, rho.^2)
@test_approx_eq Cyx * (Cxx \ Cxy) * Py scale(Cyy * Py, rho.^2)
@test_approx_eq Px qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test_approx_eq Py qnormalize!(Cyy \ (Cyx * Px), Cyy)

# Y ~ X
M = ccacov(Cyy, Cxx, Cyx, ym, xm, p)
Py = xprojection(M)
Px = yprojection(M)
rho = correlations(M)
@test xindim(M) == dy
@test yindim(M) == dx
@test outdim(M) == p
@test xmean(M) == ym
@test ymean(M) == xm
@test issorted(rho; rev=true)

@test_approx_eq Px' * Cxx * Px eye(p)
@test_approx_eq Py' * Cyy * Py eye(p)
@test_approx_eq Cxy * (Cyy \ Cyx) * Px scale(Cxx * Px, rho.^2)
@test_approx_eq Cyx * (Cxx \ Cxy) * Py scale(Cyy * Py, rho.^2)
@test_approx_eq Px qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test_approx_eq Py qnormalize!(Cyy \ (Cyx * Px), Cyy)


## ccasvd

# n > d
M = ccasvd(Zx, Zy, xm, ym, p)
Px = xprojection(M)
Py = yprojection(M)
rho = correlations(M)
@test xindim(M) == dx
@test yindim(M) == dy
@test outdim(M) == p
@test xmean(M) == xm
@test ymean(M) == ym
@test issorted(rho; rev=true)

@test_approx_eq Px' * Cxx * Px eye(p)
@test_approx_eq Py' * Cyy * Py eye(p)
@test_approx_eq Cxy * (Cyy \ Cyx) * Px scale(Cxx * Px, rho.^2)
@test_approx_eq Cyx * (Cxx \ Cxy) * Py scale(Cyy * Py, rho.^2)
@test_approx_eq Px qnormalize!(Cxx \ (Cxy * Py), Cxx)
@test_approx_eq Py qnormalize!(Cyy \ (Cyx * Px), Cyy)

# n < d
Zx_ = Zx[:, 1:4]
Zy_ = Zy[:, 1:4]
Cxx_ = cov(Zx_; vardim=2, mean=0)
Cyy_ = cov(Zy_; vardim=2, mean=0)

M = ccasvd(Zx_, Zy_, xm, ym, p)
Px = xprojection(M)
Py = yprojection(M)
@test xindim(M) == dx
@test yindim(M) == dy
@test outdim(M) == p
@test xmean(M) == xm
@test ymean(M) == ym

@test_approx_eq Px' * Cxx_ * Px eye(p)
@test_approx_eq Py' * Cyy_ * Py eye(p)


