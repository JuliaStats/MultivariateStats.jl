"""
    FactorRotationMethod

An abstract type for factor rotation methods.
"""
abstract type FactorRotationMethod end

"""
    Orthogonal

A type representing orthogonal factor rotations.
"""
struct Orthogonal <: FactorRotationMethod end

"""
    Oblique

A type representing oblique factor rotations.
"""
struct Oblique <: FactorRotationMethod end

"""
    FactorRotationCriterion{T <: FactorRotationMethod}

An abstract type for factor rotation criteria for a specific factor
rotation type.
"""
abstract type FactorRotationCriterion{T<:FactorRotationMethod} end

"""
    CrawfordFerguson{T} <: FactorRotationCriterion{T}

A type representing the Crawford-Ferguson family of factor rotation criteria.
These criteria are valid for both orthogonal and oblique rotation. The method
of rotation can be set through the type parameter `T`.

This criterion minimizes

`Q(Λ) = (1 - k) * tr((Λ.^2)' * Λ.^2 * N) / 4 + k * tr((Λ.^2)' * M * Λ.^2) / 4`

where `Λ` is a `d x p` rotated loading matrix, `N` is a `p x p`
matrix with zeros on the diagonal and ones everywhere else, `M` is an
analogous `d x d` matrix, and `k` is a non-negative shape parameter.

**Parameters**
- `kappa` is a non-negative shape parameter. In the orthogonal setting,
  some classical special cases are
  - `kappa = 0` is quartimax rotation
  - `kappa = 1 / d` is varimax rotation
  - `kappa = p / (2 * d)` is equamax rotation
  - `kappa = (p - 1) / (d + p - 2)` is parsimax rotation
  - `kappa = 1` is factor parsimony rotation

**References**
- Crawford, C.B. and Ferguson, G.A. (1970). A general rotation criterion and
  its use in orthogonal rotation. Psychometrika, 35, 321-332.
  doi 10.1007/BF02310792
- Browne, M.W. (2001). An overview of analytic rotation in exploratory factor
  analysis. Multivariate Behavioral Research, 36, 111-150.
  doi 10.1207/S15327906MBR3601_05
"""
struct CrawfordFerguson{T} <: FactorRotationCriterion{T}
    kappa::Real

    CrawfordFerguson{T}(; kappa::Union{Real,Integer}=0.0) where {T<:FactorRotationMethod} = begin
        kappa ≥ zero(eltype(kappa)) ||
        throw(DomainError("CrawfordFerguson: kappa needs to be non-negative"))

        new(float(kappa))
    end
end

function ∇Qf(L::AbstractMatrix, C::CrawfordFerguson)
    d, p = size(L)
    N = ones(eltype(L), p, p) - Matrix{eltype(L)}(I, p, p)
    M = ones(eltype(L), d, d) - Matrix{eltype(L)}(I, d, d)
    L2 = L .^ 2

    return (
        (1 - C.kappa) * L .* (L2 * N) + C.kappa * L .* (M * L2),
        (1 - C.kappa) * sum(L2 .* (L2 * N)) / 4.0 + C.kappa * sum(L2 .* (M * L2)) / 4.0
    )
end

"""
    Varimax <: FactorRotationCriterion{Orthogonal}

Convenience type for the varimax factor rotation criterion.

This criterion minimizes

`Q(Λ) = -norm(Λ.^2 .- mean.(eachcol(Λ.^2))')^2 / 4.0`

where `Λ` is a `d x p` matrix of rotated loadings.

**References**
- Kaiser, H.F. (1958). The varimax criterion for analytic rotation in factor
  analysis. Psychometrika, 23, 187-200.
- Harman, H.H. (1976). Modern factor analysis (3rd. ed.).
  Chicago: The University of Chicago Press. Page 290.
"""
struct Varimax <: FactorRotationCriterion{Orthogonal} end

function ∇Qf(L::AbstractMatrix, ::Varimax)
    Q = L .^ 2 .- mean.(eachcol(L .^ 2))'
    return (-L .* Q, -norm(Q)^2 / 4.0)
end

"""
    Quartimax <: FactorRotationCriterion{Orthogonal}

Convenience type for the quartimax factor rotation criterion.

This criterion minimizes

`Q(Λ) = -norm(Λ.^2)^2 / 4.0`

where `Λ` is a `d x p` matrix of rotated loadings.

**References**
- Carroll, J.B. (1953). An analytic solution for approximating simple
  structure in factor analysis. Psychometrika, 18, 23-38.
- Ferguson, G.A. (1954). The concept of parsimony in factor analysis.
  Psychometrika, 19, 281-290.
- Neuhaus, J.O. & Wrigley, C. (1954). The quartimax method: An analytical
  approach to orthogonal simple structure.
  British Journal of Statistical Psychology, 7, 81-91.
"""
struct Quartimax <: FactorRotationCriterion{Orthogonal} end

function ∇Qf(L::AbstractMatrix, ::Quartimax)
    Q = L .^ 2
    return (-L .* Q, -norm(Q)^2 / 4.0)
end

"""
    MinimumEntropy <: FactorRotationCriterion{Orthogonal}

A type representing a simple entropy criterion for factor rotation.

This criterion minimizes

`Q(Λ) = -tr((Λ.^2)' * log.(Λ.^2)) / 2.0`

where `Λ` is a `d x p` matrix of rotated loadings.

**Note**
No oblique version of this criterion exits.

**References**
- Jennrich, R. I. (2004). Rotation to simple loadings using component
  loss functions: The orthogonal case. Psychometrika, 69, 257-273.
  doi 10.1007/BF02295943
"""
struct MinimumEntropy <: FactorRotationCriterion{Orthogonal} end

function ∇Qf(L::AbstractMatrix, ::MinimumEntropy)
    L2 = L .^ 2

    return (
        -L .* log.(L2) - L,
        -sum(L2 .* log.(L2)) / 2.0
    )
end

"""
    Oblimin{T} <: FactorRotationCriterion{T}

Type for the oblimin family of factor rotation criteria.
These criteria are valid for both orthogonal and oblique rotation. The method
of rotation can be set through the type parameter `T`. 

This criterion minimizes

`Q(Λ) = -tr((Λ.^2)' * (I - gamma / d * C) *  Λ.^2 * N) / 4.0`

where `Λ` is a `d x p` matrix of rotated loadings, `I` is the `d`-dimensional
identity matrix, `C` is a `d x d` matrix with only ones, `N` is a `p x p`
matrix with zeros on the diagonal and ones everywhere else, and `gamma` is a
shape parameter.

**Parameters**
- `gamma` is a shape parameter. Negative values are allowed and might be useful
  for oblique rotations.
    
  In the setting of oblique factor rotation, some special cases are
  - `gamma = 0` is the quartimin criterion
  - `gamma = 1/2` is the biquartimin criterion
  - `gamma = 1` is the covarimin criterion
  
  In the setting of orthogonal factor rotation, the oblimin family of factor
  rotations is equivalent to the orthomax family of factor rotation criteria
  and some special cases are
  - `gamma = 0` is the quartimax criterion
  - `gamma = 0.5` is the biquartimax criterion
  - `gamma = 1` is the varimax criterion
  - `gamma = d / 2.0` is the equamax criterion

**References**
- Carroll, J.B. (1960). IBM 704 program for generalized analytic rotation
  solution in factor analysis. Harvard University, unpublished.
- Harman, H.H. (1976). Modern factor analysis (3rd. ed.).
  Chicago: The University of Chicago Press. Page 322.
- Jennrich, R.I. (1979). Admissible values of κ in direct oblimin rotation.
  Psychometrika, 44, 173-177.
"""
struct Oblimin{T} <: FactorRotationCriterion{T}
    γ::Real

    Oblimin{T}(; γ::Union{Real,Integer}=0.0) where {T<:FactorRotationMethod} = begin
        new(float(γ))
    end
end

function ∇Qf(L::AbstractMatrix, C::Oblimin)
    d, p = size(L)
    Q = L .^ 2 * (ones(eltype(L), p, p) - Matrix{eltype(L)}(I, p, p))
    if C.gamma != zero(eltype(C.gamma))
        Q = (Matrix{eltype(L)}(I, d, d) - C.gamma / d * ones(eltype(L), d, d)) * Q
    end
    return (L .* Q, sum(L .^ 2 .* Q) / 4.0)
end

"""
    Quartimin <: FactorRotationCriterion{Oblique}

Convenience type for the quartimin factor rotation criterion.

This criterion minimizes

`Q(Λ) = -tr((Λ.^2)' *  Λ.^2 * N) / 4.0`

where `Λ` is a `d x p` matrix of rotated loadings, and `N` is a `p x p`
matrix with zeros on the diagonal and ones everywhere else.

**References**
- Carroll, J.B. (1960). IBM 704 program for generalized analytic rotation
  solution in factor analysis. Harvard University, unpublished.
"""
struct Quartimin <: FactorRotationCriterion{Oblique} end

function ∇Qf(L::AbstractMatrix, ::Quartimin)
    _, p = size(L)
    Q = L .^ 2 * (ones(eltype(L), p, p) - Matrix{eltype(L)}(I, p, p))
    return (L .* Q, sum(L .^ 2 .* Q) / 4.0)
end

"""
    gparotate(F, C::FactorRotationCriterion{Orthogonal}; ...)

Compute an orthogonal rotation of a loading matrix according to the
provided rotation criterion. A gradient projection algorithm is used
to perform the computation

**Parameters**
- `F` is the matrix of loadings to be rotated
- `C` is a factor rotation criterion
- If `normalizerows` is true, then the rows of `F` are normalized to
  length 1 before rotation and the rows of the rotated loadings are
  scaled back.
- If `randominit` is true, then random initial values are used.
  Otherwise, the algorithm is initialized with the identity matrix.
- `maxiter` determines the maximum number of iterations
- `lsiter` determines the maximum number of iterations spent on
  line search
- `ϵ` is the convergence tolerance

**References**
- Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms
  and Software for Arbitrary Rotation Criteria in Factor Analysis.
  Educational and Psychological Measurement, 65, 676-696.
  doi 10.1177/0013164404272507
- Jennrich, R. I. (2001). A simple general procedure for orthogonal rotation.
  Psychometrika, 66, 289-306. doi 10.1007/BF02294840
"""
function gparotate(F::AbstractMatrix,
    C::FactorRotationCriterion{Orthogonal};
    normalizerows=false,
    randominit=false,
    maxiter::Integer=1000,
    lsiter::Integer=10,
    ϵ::Real=1.0e-6)
    d, p = size(F)
    if d < 2
        return (F, Matrix{eltype(F)}(I, p, p))
    end

    w = zeros(eltype(F), d)
    if normalizerows
        w = norm.(eachrow(F))
        F ./= w
    end

    # Setup
    if randominit
        T = qr(randn(eltype{F}, p, p)).Q
    else
        T = Matrix{eltype(F)}(I, p, p)
    end
    α = 1
    L = F * T

    ∇Q, f = ∇Qf(L, C)
    ∇f = F' * ∇Q

    s = 0
    for _ in 1:maxiter
        M = T' * ∇f
        S = (M + M') / 2
        ∇fp = ∇f - T * S

        # Check for convergence
        s = norm(∇fp)
        if s < ϵ
            break
        end

        α *= 2
        # Create temporaries here so they are not local to the loop
        Tt = zeros(eltype(T), p, p)
        ∇Qt, ft = ∇Q, f
        for _ in 1:lsiter
            # Line search to project the gradient step back onto
            # the Stiefel manifold
            X = T - α * ∇fp
            UDV = svd(X)
            Tt = UDV.U * UDV.Vt
            L = F * Tt
            ∇Qt, ft = ∇Qf(L, C)
            if (ft < (f - s^2 * α / 2))
                break
            end
            α /= 2
        end
        T = Tt
        f = ft
        ∇f = F' * ∇Qt
    end

    (s < ϵ) || throw(ConvergenceException(maxiter, s, ϵ))

    if normalizerows
        L .*= w
    end

    (Matrix(L), Matrix(T))
end

"""
    gparotate(F, C::FactorRotationCriterion{Oblique}; ...)

Compute an oblique rotation of a loading matrix according to the
provided rotation criterion. A gradient projection algorithm is used
to perform the computation

**Parameters**
- `F` is the matrix of loadings to be rotated
- `C` is a factor rotation criterion
- If `normalizerows` is true, then the rows of `F` are normalized to
  length 1 before rotation and the rows of the rotated loadings are
  scaled back.
- If `randominit` is true, then random initial values are used.
  Otherwise, the algorithm is initialized with the identity matrix.
- `maxiter` determines the maximum number of iterations
- `lsiter` determines the maximum number of iterations spent on
  line search
- `ϵ` is the convergence tolerance

**References**
- Bernaards, C.A. and Jennrich, R.I. (2005) Gradient Projection Algorithms
  and Software for Arbitrary Rotation Criteria in Factor Analysis.
  Educational and Psychological Measurement, 65, 676-696.
  doi 10.1177/0013164404272507
- Jennrich, R. I. (2002). A simple general method for oblique rotation.
  Psychometrika, 67, 7-19. doi 10.1007/BF02294706
"""
function gparotate(F::AbstractMatrix,
    C::FactorRotationCriterion{Oblique};
    normalizerows=false,
    randominit=false,
    maxiter::Integer=1000,
    lsiter::Integer=10,
    ϵ::Real=1.0e-6)
    d, p = size(F)
    if d < 2
        return (F, Matrix{eltype(F)}(I, p, p))
    end

    w = zeros(eltype(F), d)
    if normalizerows
        w = norm.(eachrow(F))
        F ./= w
    end

    # Setup
    if randominit
        T = randn(eltype{F}, p, p)
        T ./= norm.(eachcol(T))'
    else
        T = Matrix{eltype(F)}(I, p, p)
    end
    α = 1
    L = (T \ F')'

    ∇Q, f = ∇Qf(L, C)
    ∇f = T' \ (-∇Q' * L)

    s = 0
    for _ in 1:maxiter
        ∇fp = ∇f - T .* sum.(eachcol(T .* ∇f))'
        s = norm(∇fp)

        # Check for convergence
        if s < ϵ
            break
        end

        α *= 2
        # Create temporaries here so they are not local to the loop
        Tt = zeros(eltype(F), p, p)
        ∇Qt, ft = ∇Q, f
        for _ in 1:lsiter
            # Line search to project the gradient step back onto
            # the oblique manifold
            X = T - α * ∇fp
            v = 1 ./ norm.(eachcol(X))
            Tt = X .* v'
            L = (Tt \ F')'
            ∇Qt, ft = ∇Qf(L, C)
            if (ft < (f - s^2 * α / 2))
                break
            end
            α /= 2
        end
        T = Tt
        f = ft
        ∇f = T' \ (-∇Qt' * L)
    end

    (s < ϵ) || throw(ConvergenceException(maxiter, s, ϵ))

    if normalizerows
        L .*= w
    end

    (Matrix(L), Matrix(T))
end

"""
    FactorRotation{T <: Real}

This type contains the result of a factor rotation. The matrix `L` contains
the rotated loadings and the matrix `R` contains the rotation matrix which
was applied to the original loadings.
"""
struct FactorRotation{T<:Real}
    L::Matrix{T}
    R::Matrix{T}
end

"""
    loadings(M::FactorRotation)

Return rotated loading matrix.
"""
loadings(M::FactorRotation) = M.L

"""
    rotation(M::FactorRotation)

Return rotation matrix used to obtain rotated loadings.
"""
rotation(M::FactorRotation) = M.R

"""
    rotate(L, C::FactorRotationCriterion{T <: FactorRotationMethod}; ...)

Rotate the loadings in matrix `L` using the criterion `C`.

**Parameters**
- `L` is the matrix of loadings to be rotated
- `C` is a factor rotation criterion

**Keyword parameters**
- If `normalizerows` is true, then the rows of `F` are normalized to
  length 1 before rotation and the rows of the rotated loadings are
  scaled back.
- If `randominit` is true, then random initial values are used.
  Otherwise, the algorithm is initialized with the identity matrix.
- `maxiter` determines the maximum number of iterations
- `lsiter` determines the maximum number of iterations spent on
  line search
- `tol` is the convergence tolerance
"""
function rotate(L::AbstractMatrix,
    C::FactorRotationCriterion{T};
    normalizerows::Bool=false,
    randominit::Bool=false,
    maxiter::Integer=1000,
    lsiter::Integer=10,
    tol::Real=1.0e-6) where {T<:FactorRotationMethod}
    return FactorRotation(
        gparotate(L, C; normalizerows, randominit, maxiter, lsiter, tol)...
    )
end

"""
    rotate(M::FactorAnalysis, C::FactorRotationCriterion{T <: FactorRotationMethod}; ...)

Rotate the loadings of the Factor Analysis model `M` using the criterion `C`.
Modifies the loading matrix in `M`. Parameters as in [`rotate`](@ref).
"""
function rotate!(M::FactorAnalysis,
    C::FactorRotationCriterion{T};
    normalizerows::Bool=false,
    randominit::Bool=false,
    maxiter::Integer=1000,
    lsiter::Integer=10,
    tol::Real=1.0e-6) where {T<:FactorRotationMethod}
    FR = rotate(loadings(M), C; normalizerows, randominit, maxiter, lsiter, tol)
    M.W .= loadings(FR)

    return M
end

"""
    rotate(M::PCA, C::FactorRotationCriterion{T <: FactorRotationMethod}; ...)

Rotate the components of the PCA model `M` using the criterion `C`.
Modifies the projection matrix in `M`. Parameters as in [`rotate`](@ref).
"""
function rotate!(M::PCA,
    C::FactorRotationCriterion{T};
    normalizerows::Bool=false,
    randominit::Bool=false,
    maxiter::Integer=1000,
    lsiter::Integer=10,
    tol::Real=1.0e-6) where {T<:FactorRotationMethod}
    FR = rotate(projection(M), C; normalizerows, randominit, maxiter, lsiter, tol)
    M.proj .= loadings(FR)

    return M
end