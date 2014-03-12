using MultivariateAnalysis
using Base.Test

import MultivariateAnalysis: pca_prepare, compute_cov, pca_decide_outdim

srand(34568)

## testing data

d = 5
n = 1000

R = qr(randn(d, d))[1]
@test_approx_eq R'R eye(5)
scale!(R, rand(d) .+ 0.5)

center = randn(5)

Z = R'randn(5, n)
X = Z .+ center
w = rand(n)

Xcopy = copy(X)  

### test the correctness of supporting functions

# pca_prepare

@test_approx_eq pca_prepare(X, Float64[], Float64[]) X
@test_approx_eq pca_prepare(X, Float64[], center)    X .- center
@test_approx_eq pca_prepare(X, w, Float64[])         scale(X, sqrt(w))
@test_approx_eq pca_prepare(X, w, center)            scale(X .- center, sqrt(w))

# compute_cov

@test_approx_eq compute_cov(Z, 5.0)  (Z * Z') ./ 5.0

# pca_decide_outdim

evs = [0.4, 0.3, 0.2, 0.1, 1.0e-6]
s = sum(evs)
@test pca_decide_outdim(evs, s, 3, 1.00, 1.0e-8) == 3
@test pca_decide_outdim(evs, s, 5, 0.95, 1.0e-8) == 4
@test pca_decide_outdim(evs, s, 5, 0.89, 1.0e-8) == 3
@test pca_decide_outdim(evs, s, 5, 1.00, 1.0e-4) == 4
@test pca_decide_outdim(evs, s, 3, 0.95, 1.0e-8) == 3

### test pcacov

C = (Z * Z') ./ n
es = eigvals(C)
U = eigvecs(C)
si = sortperm(es; rev=true)
es = es[si]
U = U[:,si]

# zero center

m = pcacov(C; maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test isempty(m.center)
@test_approx_eq m.projection U[:,1:3]
@test_approx_eq m.projection'm.projection eye(3)
@test_approx_eq m.principalvars es[1:3] 
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)
@test_approx_eq m.cumratios cumsum(m.principalvars) ./ m.totalvar

Y = transform(m, Z)
Zr = reconstruct(m, Y)
@test size(Y) == (3, n)
@test size(Zr) == (d, n)
@test_approx_eq Y m.projection'Z
@test_approx_eq mean(Y.^2, 2) m.principalvars
@test_approx_eq abs2(vecnorm(Zr - Z)) / n m.residualvar

@test_approx_eq transform(m, Z[:,1]) Y[:,1]
@test_approx_eq reconstruct(m, Y[:,1]) Zr[:,1]

# nonzero center

m = pcacov(C; center=center, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test m.center === center
@test_approx_eq m.projection U[:,1:3]
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)
@test_approx_eq m.cumratios cumsum(m.principalvars) ./ m.totalvar

Y = transform(m, X)
Xr = reconstruct(m, Y)
@test size(Y) == (3, n)
@test size(Xr) == (d, n)
@test_approx_eq Y m.projection'Z
@test_approx_eq mean(Y.^2, 2) m.principalvars
@test_approx_eq abs2(vecnorm(Xr - X)) / n m.residualvar

@test_approx_eq transform(m, X[:,1]) Y[:,1]
@test_approx_eq reconstruct(m, Y[:,1]) Xr[:,1]


### test pcasvd

m = pcasvd(Z, float64(n); maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test isempty(m.center)
@test_approx_eq abs(m.projection) abs(U[:,1:3])
@test_approx_eq m.projection'm.projection eye(3)
@test_approx_eq m.principalvars es[1:3]

m = pcasvd(Z, float64(n); center=center, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test m.center === center
@test_approx_eq abs(m.projection) abs(U[:,1:3])
@test_approx_eq m.projection'm.projection eye(3)
@test_approx_eq m.principalvars es[1:3]


### test pca function

# non-weighted

mv = mean(X, 2)
Z = X .- mv
C = (1/n) * (Z * Z')
ef = eigfact(C)
es = ef.values
U = ef.vectors
si = sortperm(es; rev=true)
es = es[si]
U = U[:, si]

m = pca(X; maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test_approx_eq m.center mv
@test_approx_eq m.projection U[:,1:3]
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)

Y = transform(m, X)
Xr = reconstruct(m, Y)
@test size(Y) == (3, n)
@test size(Xr) == (d, n)
@test_approx_eq Y m.projection'Z
@test_approx_eq mean(Y.^2, 2) m.principalvars


m = pca(Z; zerocenter=true, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test isempty(m.center)
@test_approx_eq m.projection U[:,1:3]
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)

m = pca(Z; method=:svd, zerocenter=true, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test isempty(m.center)
@test_approx_eq abs(m.projection) abs(U[:,1:3])
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)

# weighted

mv = sum(X .* reshape(w, 1, n), 2) ./ sum(w)
Z = X .- mv
C = (1/sum(w)) * (scale(Z, w) * Z')
C = 0.5 * (C + C')
ef = eigfact(C)
es = ef.values
U = ef.vectors
si = sortperm(es; rev=true)
es = es[si]
U = U[:, si]

m = pca(X; weights=w, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test_approx_eq m.center mv
@test_approx_eq m.projection U[:,1:3]
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)

m = pca(X; weights=w, method=:svd, maxoutdim=3)
@test isa(m, PCA)
@test indim(m) == d
@test outdim(m) == 3
@test_approx_eq m.center mv
@test_approx_eq abs(m.projection) abs(U[:,1:3])
@test_approx_eq m.principalvars es[1:3]
@test_approx_eq m.residualvar sum(es[4:5])
@test_approx_eq m.totalvar trace(C)


