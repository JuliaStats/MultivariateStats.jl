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

centers = [zeros(5) [10.0;zeros(4)] [0.0;10.0;zeros(3)]]

# Case 1: 3 groups of 500
dX = randn(5,1500);
for i = 0:500:1000
    dX[:,(1:500)+i] = dX[:,(1:500)+i] .- mean(dX[:,(1:500)+i],2)  # make the mean of each 0
end
dX1 = dX
X1 = [dX[:,1:500].+centers[:,1] dX[:,501:1000].+centers[:,2] dX[:,1001:1500].+centers[:,3]]
label1 = [fill(1,500); fill(2,500); fill(3,500)]
# Case 2: 3 groups, one with 1000, one with 100, and one with 10
dX = randn(5,1110);
dX[:,   1:1000] = dX[:,   1:1000] .- mean(dX[:,   1:1000],2)
dX[:,1001:1100] = dX[:,1001:1100] .- mean(dX[:,1001:1100],2)
dX[:,1101:1110] = dX[:,1101:1110] .- mean(dX[:,1101:1110],2)
dX2 = dX
X2 = [dX[:,1:1000].+centers[:,1] dX[:,1001:1100].+centers[:,2] dX[:,1101:1110].+centers[:,3]]
label2 = [fill(1,1000); fill(2,100); fill(3,10)]

for (X, dX, label) in ((X1, dX1, label1), (X2, dX2, label2))
    w = [length(find(label.==1)), length(find(label.==2)), length(find(label.==3))]
    M = fit(SubspaceLDA, X, label)
    @test indim(M) == 5
    @test outdim(M) == 2
    totcenter = vec(sum(centers.*w',2)./sum(w))
    @test_approx_eq mean(M) totcenter
    @test_approx_eq classmeans(M) centers
    @test classweights(M) == w
    x = rand(5)
    @test_approx_eq transform(M, x) projection(M)'*x
    dcenters = centers .- totcenter
    Hb = dcenters.*sqrt(w)'
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
end

# High-dimensional case (undersampled => singularities)
X = randn(10^6, 9)
label = rand(1:3, 9); label[1:3] = 1:3
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

# Test normalized LDA
function gen_ldadata_2(centers, n1, n2)
    d = size(centers, 1)
    X = randn(d, n1+n2)
    X[:,1:n1]       .-= vec(mean(X[:,1:n1],2))
    X[:,n1+1:n1+n2] .-= vec(mean(X[:,n1+1:n1+n2],2))
    dX = copy(X)
    X[:,1:n1]       .+= centers[:,1]
    X[:,n1+1:n1+n2] .+= centers[:,2]
    label = [fill(1,n1); fill(2,n2)]
    X, dX, label
end
centers = zeros(2,2); centers[1,2] = 10
n1 = 100
for n2 in (100, 1000, 10000)
    X, dX, label = gen_ldadata_2(centers, n1, n2)
    M = fit(SubspaceLDA, X, label; normalize=true)
    proj = projection(M)
    Hb = centers .- mean(centers, 2)   # not weighted by number in each cluster
    Sb = Hb*Hb'
    Sw = dX*dX'/(n1+n2)
    @test_approx_eq Sb*proj Sw*proj*Diagonal(M.λ)
end
