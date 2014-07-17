using MultivariateStats
using Base.Test

srand(34568)

## LinearDiscriminant

@assert LinearDiscriminant <: Discriminant

w = [1., 2., 3., 4., 5.]
b = 2.5
x = [4., 5., 2., 3., 1.]

f = LinearDiscriminant(w, b)
@test length(f) == 5
@test evaluate(f, x) == 39.5
@test evaluate(f, -x) == -34.5

@test predict(f, x) == true
@test predict(f, -x) == false

X = rand(5, 8)
Y = evaluate(f, X)
@test size(Y) == (8,)
for i = 1:8
	@test_approx_eq Y[i] evaluate(f, X[:,i])
end
@test predict(f, X) == (Y .> 0)

## prepare data

t1 = deg2rad(30)
t2 = deg2rad(75)

R1 = [cos(t1) -sin(t1); sin(t1) cos(t1)]
R2 = [cos(t2) -sin(t2); sin(t2) cos(t2)]

n = 20
Xp = scale([1.2, 3.6], randn(2, n)) .+ [1.0, -3.0]
Xn = scale([2.8, 1.8], randn(2, n)) .+ [-5.0, 2.0]

up = vec(mean(Xp, 2))
un = vec(mean(Xn, 2))
Cp = cov(Xp; vardim=2)
Cn = cov(Xn; vardim=2)
C = 0.5 * (Cp + Cn)

w_gt = C \ (up - un)
w_gt .*= (2 / (dot(w_gt, up) - dot(w_gt, un)))
b_gt = 1.0 - dot(w_gt, up)

@test_approx_eq dot(w_gt, up) + b_gt 1.0
@test_approx_eq dot(w_gt, un) + b_gt -1.0

## LDA

f = ldacov(C, up, un)
@test_approx_eq f.w w_gt
@test_approx_eq f.b b_gt

f = ldacov(Cp, Cn, up, un)
@test_approx_eq f.w w_gt
@test_approx_eq f.b b_gt

f = fit(LinearDiscriminant, Xp, Xn)
@test_approx_eq f.w w_gt
@test_approx_eq f.b b_gt

