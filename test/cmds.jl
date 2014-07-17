using MultivariateStats
using Base.Test

## conversion between dmat and gram 

X = randn(3, 5)
S = sumabs2(X, 1)
G0 = X'X
D0 = S .+ S' - 2 * G0
D0[diagind(D0)] = 0.0
D0 = sqrt(D0)

@assert issym(D0)
@assert issym(G0)

D = gram2dmat(G0)
@test issym(D)
@test_approx_eq D D0

G = dmat2gram(D0)
@test issym(G)
@test_approx_eq gram2dmat(G) D0

