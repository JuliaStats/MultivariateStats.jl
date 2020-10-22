"""
    FactorRotationAlgorithm

An abstract type for factor rotation algorithms.
"""
abstract type FactorRotationAlgorithm end

"""
    Orthomax <: FactorRotationAlgorithm

A type representing the orthomax factor rotation algorithm.

The positive parameter `γ::Real` determines which type of rotation is
performed. For a `n x p` matrix, the default `γ = 1.0` leads to varimax
rotation, `γ = 0.0` is quartimax rotation, `γ = p / 2` is equamax
rotation, and `γ = n (p - 1) / (n + p - 2)` is parsimax rotation.

The parameter `maxiter::Integer` controls the maximum number of iterations
to perform (default `1000`), `miniter::Integer` controls the minimum number
of iterations taken before convergence is checked, and `ϵ::Real` is a small
positive constant determining convergence (default `1e-12`).
"""
struct Orthomax <: FactorRotationAlgorithm
    γ::Real
    miniter::Integer
    maxiter::Integer
    ϵ::Real

    Orthomax(;γ::Union{Real, Integer} = 1.0,
              miniter::Integer = 20,
              maxiter::Integer = 1000,
              ϵ::Real = 1e-12) = begin
        γ ≥ zero(eltype(γ)) ||
            throw(DomainError("Orthomax: γ needs to be non-negative"))
        miniter > zero(eltype(miniter)) ||
            throw(DomainError("Orthomax: miniter needs to be positive"))
        maxiter > zero(eltype(maxiter)) ||
            throw(DomainError("Orthomax: maxiter needs to be positive"))
        ϵ > zero(eltype(ϵ)) ||
            throw(DomainError("Orthomax: ϵ needs to be positive"))

        new(float(γ), miniter, maxiter, ϵ)
    end
end

"""
    Varimax() -> Orthomax

Creates an orthomax factor rotation algorithm object for a matrix of
size `n x p` with `γ = 1.0`.
Remaining keyword parameters as for [`Orthomax`](@ref).
"""
function Varimax(;miniter::Integer = 20,
                  maxiter::Integer = 1000,
                  ϵ::Real = 1e-12)
    Orthomax(γ = 1.0, miniter = miniter, maxiter = maxiter, ϵ = ϵ)
end

"""
    Quartimax() -> Orthomax

Creates an orthomax factor rotation algorithm object for a matrix of
size `n x p` with `γ = 0.0`.
Remaining keyword parameters as for [`Orthomax`](@ref).
"""
function Quartimax(;miniter::Integer = 20,
                    maxiter::Integer = 1000,
                    ϵ::Real = 1e-12)
    Orthomax(γ = 0.0, miniter = miniter, maxiter = maxiter, ϵ = ϵ)
end

"""
    Equamax(p) -> Orthomax

Creates an orthomax factor rotation algorithm object for a matrix of
size `n x p` with `γ = p / 2`.
Remaining keyword parameters as for [`Orthomax`](@ref).
"""
function Equamax(p::Integer; miniter::Integer = 20,
                             maxiter::Integer = 1000,
                             ϵ::Real = 1e-12)
    Orthomax(γ = float(p) / 2.0, miniter = miniter, maxiter = maxiter, ϵ = ϵ)
end

"""
    Parsimax(n, p) -> Orthomax

Creates an orthomax factor rotation algorithm object for a matrix of
size `n x p` with `γ = n (p - 1) / (n + p - 2)`.
Remaining keyword parameters as for [`Orthomax`](@ref).
"""
function Parsimax(n::Integer, p::Integer; miniter::Integer = 20,
                                          maxiter::Integer = 1000,
                                          ϵ::Real = 1e-12)
    Orthomax(γ = float(n) * (float(p) - 1) / (float(n) + float(p) - 2),
             miniter = miniter, maxiter = maxiter, ϵ = ϵ)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, alg::Orthomax)
    summary(io, alg); println(io)
    println(io, "γ = $(alg.γ)")
    println(io, "miniter = $(alg.miniter)")
    println(io, "maxiter = $(alg.maxiter)")
    println(io, "ϵ = $(alg.ϵ)")
end

"""
    FactorRotation{T <: Real}

The return type for factor rotations.

`F` contains the rotated factors and `R` is the rotation matrix that
was applied to the original factors.
"""
struct FactorRotation{T <: Real, Ta <: FactorRotationAlgorithm}
    F::Matrix{T}
    R::Matrix{T}

    alg::Ta

    FactorRotation{T, Ta}(F, R, alg) where {T <: Real, Ta <: FactorRotationAlgorithm} = new{T, Ta}(F, R, alg)
end

eltype(::Type{FactorRotation{T, Ta}}) where {T,Ta} = T

FactorRotation(F::Matrix{T}, R::Matrix{T}, alg::Ta) where {T <: Real, Ta <: FactorRotationAlgorithm} = FactorRotation{T, Ta}(F, R, alg)
FactorRotation(F::Matrix{T}, alg::Ta) where {T <: Real, Ta <: FactorRotationAlgorithm} = FactorRotation(F, Matrix{eltype(F)}(I, size(F, 2), size(F, 2)), alg)

function FactorRotation(F::Matrix, R::Matrix, alg::Ta) where {Ta <: FactorRotationAlgorithm}
    return FactorRotation(promote(F, R)..., alg)
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, FR::FactorRotation{<:Any,<:Any})
    summary(io, FR); println(io)
end

"""
    loadings(FR::FactorRotation)

Returns the loading matrix.
"""
loadings(FR::FactorRotation{T, Ta}) where {T, Ta} = FR.F

"""
    rotation(FR::FactorRotation)

Returns the rotation matrix.
"""
rotation(FR::FactorRotation{T, Ta}) where {T, Ta} = FR.R

## Comparison to other implementations
# The implementation of varimax in R row-normlises the input matrix before
# application of the algorithm and rescales the rows afterwards.
## Reference
# Mikkel B. Stegmann, Karl Sjöstrand, Rasmus Larsen, "Sparse modeling of
# landmark and texture variability using the orthomax criterion,"
# Proc. SPIE 6144, Medical Imaging 2006: Image Processing, 61441G
# (10 March 2006); doi: 10.1117/12.651293
function orthomax(F::AbstractMatrix, γ, miniter, maxiter, ϵ)
    n, p = size(F)
    if n < 2
        return (F, Matrix{eltype(F)}(I, p, p))
    end

    # Warm up step
    # Do one step. If that first step did not lead away from the identity
    # matrix enough use a random orthogonal matrix as a starting point.
    M = svd(F' * (F .^ 3 - γ / n * F * Diagonal(vec(sum(F .^ 2, dims=1)))))
    R = M.U * M.Vt
    if norm(R - Matrix{eltype(R)}(I, p, p)) < ϵ
        R = qr(randn(p, p)).Q
    end

    # Main iterations
    d = 0
    lastchange = NaN
    converged = false
    for i in 1:maxiter
        dold  = d
        B = F * R
        M = svd(F' * (B .^ 3 - γ / n * B * Diagonal(vec(sum(B .^ 2, dims=1)))))
        R = M.U * M.Vt
        d = sum(M.S)
        lastchange = abs(d - dold) / d
        if lastchange < ϵ && i >= miniter
            converged = true
            break
        end
    end

    converged || throw(ConvergenceException(maxiter, lastchange, ϵ))

    (F * R, R)
end

"""
    fit(::Type{FactorRotation}, F::AbstractMatrix; ...) -> FactorRotation

Fit a factor rotation to the matrix `F` and apply it. The algorithm used to
perform the factor rotation is by default [`Orthomax`](@ref) and can be changed
with the keyword argument `alg` which is of type `<:FactorRotationAlgorithm`.
"""
function fit(::Type{FactorRotation}, F::AbstractMatrix;
             alg::T = Orthomax()) where {T <: FactorRotationAlgorithm}
    if isa(alg, Orthomax)
        F, R = orthomax(F, alg.γ, alg.miniter, alg.maxiter, alg.ϵ)
        return FactorRotation(F, R, alg)
    end
end

"""
    fit(::Type{FactorRotation}, F::FactorAnalysis; ...) -> FactorRotation

Fit a factor rotation to the loading matrix of the  [`FactorAnalysis`](@ref)
object and apply it.

See [`fit(::Type{FactorRotation}, F::AbstractMatrix)`](@ref) for keyword arguments.
"""
function fit(::Type{FactorRotation}, F::FactorAnalysis;
             alg::T = Orthomax()) where {T <: FactorRotationAlgorithm}
    return fit(FactorRotation, F.W; alg = alg)
end

## Alternative interface

"""
    rotatefactors(F::AbstractMatrix, [alg::FactorRotationAlgorithm]) -> FactorRotation

Rotate the factors in the matrix `F`. The algorithm to be used can be passed in
via the second argument `alg`. By default [`Orthomax`](@ref) is used.
"""
function rotatefactors(F::AbstractMatrix, alg::FactorRotationAlgorithm = Orthomax())
    F, R = _rotatefactors(F, alg)
    return FactorRotation(F, R, alg)
end

# Use multiple dispatch to decide on the algorithm implementation

function _rotatefactors(F::AbstractMatrix, alg::Orthomax)
    return orthomax(F, alg.γ, alg.miniter, alg.maxiter, alg.ϵ)
end

"""
    rotatefactors(F::FactorAnalysis, [alg::FactorRotationAlgorithm]) -> FactorAnalysis

Rotate the factors in the loading matrix of `F` which is of type [`FactorAnalysis`](@ref).
The algorithm to be used can be passed in via the second argument `alg`.
By default [`Orthomax`](@ref) is used.
"""
function rotatefactors(F::FactorAnalysis, alg::FactorRotationAlgorithm = Orthomax())
    FR = rotatefactors(F.W, alg)
    return FactorAnalysis(F.mean, FR.F, F.Ψ)
end
