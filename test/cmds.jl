using MultivariateStats
using Base.Test

## testing data

function pwdists(X)
	S = Base.sumabs2(X, 1)
	D2 = S .+ S' - 2 * (X'X)
	D2[diagind(D2)] = 0.0
	return sqrt(D2)
end

X0 = randn(3, 5)
G0 = X0'X0
D0 = pwdists(X0)

## conversion between dmat and gram

@assert issym(D0)
@assert issym(G0)

D = gram2dmat(G0)
@test issym(D)
@test_approx_eq D D0

G = dmat2gram(D0)
@test issym(G)
@test_approx_eq gram2dmat(G) D0

## classical MDS

X = classical_mds(D0, 3)
@test_approx_eq pwdists(X) D0

#Test MDS embeddings in dimensions >= number of points
@test classical_mds([0 1; 1 0], 2, dowarn=false) == [-0.5 0.5; 0 0]
@test classical_mds([0 1; 1 0], 3, dowarn=false) == [-0.5 0.5; 0 0; 0 0]

