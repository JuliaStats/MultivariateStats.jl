using MultivariateStats
using Base.Test

srand(34568)

## prepare data

d = 5
ns = [10, 15, 20]
nc = length(ns)
n = sum(ns)
Xs = Matrix{Float64}[]
ys = Vector{Int}[]
Ss = Matrix{Float64}[]
cmeans = zeros(d, nc)

for k = 1:nc
	R = qr(randn(d, d))[1]
	nk = ns[k]

	Xk = R * scale(2 * rand(d) + 0.5, randn(d, nk)) .+ randn(d)
	yk = fill(k, nk)
	uk = vec(mean(Xk, 2))
	Zk = Xk .- uk
	Sk = Zk * Zk'

	push!(Xs, Xk)
	push!(ys, yk)
	push!(Ss, Sk)
	cmeans[:,k] = uk
end

X = hcat(Xs...)
y = vcat(ys...)
mv = vec(mean(X, 2))
Sw = zeros(d, d)
for k = 1:nc
	Sw += Ss[k]
end

Sb = zeros(d, d)
for k = 1:nc
	dv = cmeans[:,k] - mv
	Sb += ns[k] * (dv * dv')
end

Z = X .- mv
Sa = Z * Z'
@test_approx_eq Sw + Sb Sa

@assert size(X) == (d, n)
@assert size(y) == (n,)

## Stats

S = multiclass_lda_stats(nc, X, y)

@test S.dim == d
@test S.nclasses == nc

@test classweights(S) == ns
@test_approx_eq classmeans(S) cmeans
@test_approx_eq mean(S) mv

@test_approx_eq withclass_scatter(S) Sw 
@test_approx_eq betweenclass_scatter(S) Sb

