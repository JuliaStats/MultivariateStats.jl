using MultivariateStats
using Base.Test

if !isdefined(Base, :normalize)
    normalize(x) = x/norm(x)
end

#Test equivalence of eigenvectors/singular vectors taking into account possible
#phase (sign) differences
#This may end up in Base.Test; see JuliaLang/julia#10651
function test_approx_eq_vecs{S<:Real,T<:Real}(a::StridedVecOrMat{S},
    b::StridedVecOrMat{T}, error=nothing)
    n = size(a, 2)
    @test n==size(b, 2) && size(a, 1)==size(b, 1)
    error==nothing && (error=n^3*(eps(S)+eps(T)))
    for i=1:n
        ev1, ev2 = a[:, i], b[:, i]
        deviation = min(abs(norm(ev1-ev2)), abs(norm(ev1+ev2)))
        if !isnan(deviation)
            @test_approx_eq_eps deviation 0.0 error
        end
    end
end

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

	Xk = R * Diagonal(2 * rand(d) + 0.5) * randn(d, nk) .+ randn(d)
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

## Solve

emax = maximum(eigvals(Sw))
Sw_c = copy(Sw)
Sb_c = copy(Sb)

lambda = 1.0e-3
Sw_r = Sw + (lambda * emax) * eye(d)

P1 = mclda_solve(Sb, Sw, :gevd, nc-1, lambda)
@test_approx_eq P1' * Sw_r * P1 eye(nc-1)
U = Sb * P1
V = Sw_r * P1
# test whether U is proportional to V,
# which indicates P is the generalized eigenvectors
@test_approx_eq U V*Diagonal(vec(mean(U./V, 1)))

P2 = mclda_solve(Sb, Sw, :whiten, nc-1, lambda)
@test_approx_eq P2' * Sw_r * P2 eye(nc-1)

test_approx_eq_vecs(P1, P2)


## LDA

M = fit(MulticlassLDA, nc, X, y; method=:gevd, regcoef=lambda)
@test indim(M) == d
@test outdim(M) == nc - 1
@test_approx_eq projection(M) P1
@test_approx_eq M.pmeans M.proj'cmeans
@test_approx_eq transform(M, X) M.proj'X

M = fit(MulticlassLDA, nc, X, y; method=:whiten, regcoef=lambda)
@test indim(M) == d
@test outdim(M) == nc - 1
# @test_approx_eq projection(M) P2  # signs may change
@test_approx_eq M.pmeans M.proj'cmeans
@test_approx_eq transform(M, X) M.proj'X


## High-dimensional LDA (subspace LDA)

# Low-dimensional case (no singularities)

dX = randn(5,1500);
# 3 groups of 500 with zero mean
for i = 0:500:1000
    dX[:,(1:500)+i] = dX[:,(1:500)+i] .- mean(dX[:,(1:500)+i],2)
end
centers = [zeros(5) [10.0;zeros(4)] [0.0;10.0;zeros(3)]]
X = [dX[:,1:500].+centers[:,1] dX[:,501:1000].+centers[:,2] dX[:,1001:1500].+centers[:,3]]
label = [fill(1,500); fill(2,500); fill(3,500)]
M = fit(SubspaceLDA, X, label)
@test indim(M) == 5
@test outdim(M) == 2
@test_approx_eq mean(M) vec(mean(centers,2))
@test_approx_eq classmeans(M) centers
@test classweights(M) == [500,500,500]
x = rand(5)
@test_approx_eq transform(M, x) projection(M)'*x
dcenters = centers .- mean(centers,2)
Hb = dcenters*sqrt(500)
Sb = Hb*Hb'
Sw = dX*dX'
# When there are no singularities, we need to solve the "regular" LDA eigenvalue equation
proj = projection(M)
@test_approx_eq Sb*proj Sw*proj*Diagonal(M.λ)
@test_approx_eq proj'*Sw*proj eye(2,2)

# also check that this is consistent with the conventional algorithm
Mld = fit(MulticlassLDA, 3, X, label)
for i = 1:2
    @test_approx_eq abs(dot(normalize(proj[:,i]), normalize(Mld.proj[:,i]))) 1
end

# High-dimensional case (undersampled => singularities)
X = randn(10^6, 9)
label = rand(1:3, 9)
M = fit(SubspaceLDA, X, label)
centers = M.cmeans
for i = 1:3
    flag = label.==i
    @test_approx_eq centers[:,i] mean(X[:,flag], 2)
    @test M.cweights[i] == sum(flag)
end
dcenters = centers .- mean(X, 2)
Hb = dcenters*Diagonal(sqrt(M.cweights))
Hw = X - centers[:,label]
@test_approx_eq (M.projw'*Hb)*(Hb'*M.projw)*M.projLDA (M.projw'*Hw)*(Hw'*M.projw)*M.projLDA*Diagonal(M.λ)
Gw = projection(M)'*Hw
@test_approx_eq Gw*Gw' eye(2,2)

# Test that nothing breaks if one class has no members
label[1:4] = 1
label[5:9] = 3  # no data points have label == 2
M = fit(SubspaceLDA, X, label)
centers = M.cmeans
dcenters = centers .- mean(X, 2)
Hb = dcenters*Diagonal(sqrt(M.cweights))
Hw = X - centers[:,label]
@test_approx_eq (M.projw'*Hb)*(Hb'*M.projw)*M.projLDA (M.projw'*Hw)*(Hw'*M.projw)*M.projLDA*Diagonal(M.λ)
