using MultivariateStats
using Base.Test

srand(34568)

## data

m = 9
n = 6
n2 = 3

X = randn(m, n)
A = randn(n, n2)
Xt = X'

b = randn(1, n2)

E = randn(m, n2) * 0.1
Y0 = X * A + E
Y1 = X * A .+ b + E

y0 = Y0[:,1]
y1 = Y1[:,1]

## llsq

A = llsq(X, Y0; trans=false, bias=false)
A_r = copy(A)
@test size(A) == (n, n2)
@test_approx_eq X'Y0 X'X * A

a = llsq(X, y0; trans=false, bias=false)
@test size(a) == (n,)
@test_approx_eq a A[:,1]

A = llsq(Xt, Y0; trans=true, bias=false)
@test size(A) == (n, n2)
@test_approx_eq A A_r

a = llsq(Xt, y0; trans=true, bias=false)
@test size(a) == (n,)
@test_approx_eq a A[:,1]

Aa = llsq(X, Y1; trans=false, bias=true)
Aa_r = copy(Aa)
@test size(Aa) == (n+1, n2)
A, b = Aa[1:end-1,:], Aa[end,:]
@test_approx_eq X'Y1 X' * (X * A .+ b)

aa = llsq(X, y1; trans=false, bias=true)
@test_approx_eq aa Aa[:,1]

Aa = llsq(Xt, Y1; trans=true, bias=true)
@test_approx_eq Aa Aa_r

aa = llsq(Xt, y1; trans=true, bias=true)
@test_approx_eq aa Aa[:,1]

