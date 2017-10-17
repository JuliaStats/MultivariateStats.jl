using MultivariateStats
using Base.Test
import StatsBase

srand(15678)

## icagfun

f = icagfun(:tanh)
u, v = evaluate(f, 1.5)
@test u ≈ 0.905148253644866438242
@test v ≈ 0.180706638923648530597

f = icagfun(:tanh, Float32)
u, v = evaluate(f, 1.5f0)
@test u ≈ 0.90514827f0
@test v ≈ 0.18070662f0

f = icagfun(:tanh, 1.5)
u, v = evaluate(f, 1.2)
@test u ≈ 0.946806012846268289646
@test v ≈ 0.155337561057228069719

f = icagfun(:tanh, 1.5f0)
u, v = evaluate(f, 1.2f0)
@test u ≈ 0.94680610f0
@test v ≈ 0.15533754f0

f = icagfun(:gaus)
u, v = evaluate(f, 1.5)
@test u ≈ 0.486978701037524594696
@test v ≈ -0.405815584197937162246

f = icagfun(:gaus, Float32)
u, v = evaluate(f, 1.5f0)
@test u ≈ 0.4869787f0
@test v ≈ -0.40581557f0


## data

# sources
n = 1000
k = 3
m = 8

t = linspace(0.0, 10.0, n)
s1 = sin.(t * 2)
s2 = s2 = 1.0 - 2.0 * Bool[isodd(floor(Int, x / 3)) for x in t]
s3 = Float64[mod(x, 5.0) for x in t]

s1 += 0.1 * randn(n)
s2 += 0.1 * randn(n)
s3 += 0.1 * randn(n)

S = hcat(s1, s2, s3)'
@assert size(S) == (k, n)

A = randn(m, k)

X = A * S
mv = vec(mean(X,2))
@assert size(X) == (m, n)
C = cov(X, 2)

# FastICA

M = fit(ICA, X, k; do_whiten=false, tol=Inf)
@test isa(M, ICA)
@test indim(M) == m
@test outdim(M) == k
@test mean(M) == mv
W = M.W
@test transform(M, X) ≈ W' * (X .- mv)
@test W'W ≈ eye(k)

M = fit(ICA, X, k; do_whiten=true, tol=Inf)
@test isa(M, ICA)
@test indim(M) == m
@test outdim(M) == k
@test mean(M) == mv
W = M.W
@test W'C * W ≈ eye(k)

@test_throws StatsBase.ConvergenceException fit(ICA, X, k; do_whiten=true, tol=1)

# Use data of different type

XX = convert(Matrix{Float32}, X)

M = fit(ICA, XX, k; do_whiten=true, tol=Inf)
@test eltype(mean(M)) == Float32
@test eltype(M.W) == Float32

M = fit(ICA, XX, k; do_whiten=false, tol=Inf)
@test isa(M, ICA)
@test eltype(mean(M)) == Float32
@test eltype(M.W) == Float32
W = M.W
@test transform(M, X) ≈ W' * convert(Matrix{Float32}, (X .- mv))
@test W'W ≈ eye(k)
