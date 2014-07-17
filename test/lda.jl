using MultivariateStats
using Base.Test

## LinearDiscriminant

@assert LinearDiscriminant <: Discriminant

w = [1., 2., 3., 4., 5.]
b = 2.5
x = [4., 5., 2., 3., 1.]

f = LinearDiscriminant(w, b)
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


