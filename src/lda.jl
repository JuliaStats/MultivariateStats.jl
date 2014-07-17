# Linear Discriminant Analysis

#### Type to represent a linear discriminant functional

abstract Discriminant

immutable LinearDiscriminant <: Discriminant
    w::Vector{Float64}
    b::Float64
end

evaluate(f::LinearDiscriminant, x::AbstractVector) = dot(f.w, x) + f.b

function evaluate(f::LinearDiscriminant, X::AbstractMatrix)
    R = At_mul_B(X, f.w)
    if f.b != 0
        broadcast!(+, R, R, f.b)
    end
    return R
end

predict(f::Discriminant, x::AbstractVector) = evaluate(f, x) > 0

predict(f::Discriminant, X::AbstractMatrix) = (Y = evaluate(f, X); Bool[y > 0 for y in Y])


#### function to solve linear discriminant

function ldacov(C::AbstractMatrix{Float64}, 
                μp::AbstractVector{Float64}, 
                μn::AbstractVector{Float64})

    w = cholfact(C) \ (μp - μn)
    ap = dot(w, μp)
    an = dot(w, μn)
    c = 2 / (ap - an)
    LinearDiscriminant(scale!(w, c), 1 - c * ap)
end

ldacov(Cp::AbstractMatrix{Float64}, 
       Cn::AbstractMatrix{Float64}, 
       μp::AbstractVector{Float64}, 
       μn::AbstractVector{Float64}) = ldacov(Cp + Cn, μp, μn)

#### interface functions

function fit(::Type{LinearDiscriminant}, Xp::Matrix{Float64}, Xn::Matrix{Float64})
    μp = vec(mean(Xp, 2))
    μn = vec(mean(Xn, 2))
    Zp = Xp .- μp
    Zn = Xn .- μn
    Cp = A_mul_Bt(Zp, Zp)
    Cn = A_mul_Bt(Zn, Zn)
    ldacov(Cp, Cn, μp, μn)
end

