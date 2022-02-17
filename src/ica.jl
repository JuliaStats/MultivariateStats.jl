# Independent Component Analysis

"""
This type contains ICA model parameters: mean and component matrix ``W``.

**Note:** Each column of the component matrix ``W`` corresponds to an independent component.
"""
struct ICA{T<:Real} <: LinearDimensionalityReduction
    mean::Vector{T}   # mean vector, of length m (or empty to indicate zero mean)
    W::Matrix{T}      # component coefficient matrix, of size (m, k)
end
function show(io::IO, M::ICA)
    indim, outdim = size(M)
    print("ICA(indim=$indim, outdim=$outdim)")
end

"""
    size(M::ICA)

Returns a tuple with the input dimension, *i.e* the number of observed mixtures, and
the output dimension, *i.e* the number of independent components.
"""
size(M::ICA) = size(M.W)

"""
    mean(M::ICA)

Returns the mean vector.
"""
mean(M::ICA) = fullmean(size(M.W,1), M.mean)

"""
    predict(M::ICA, x)

Transform `x` to the output space to extract independent components,
as ``\\mathbf{W}^T (\\mathbf{x} - \\boldsymbol{\\mu})``, given the model `M`.
"""
predict(M::ICA, x::AbstractVecOrMat{<:Real}) = transpose(M.W) * centralize(x, M.mean)


#### core algorithm

"""
The abstract type for all `g` (derivative) functions.

Let `g` be an instance of such type, then `update!(g, U, E)` given

- `U = w'x`

returns updated in-place `U` and `E`, s.t.

- `g(w'x) --> U` and `E{g'(w'x)} --> E`

"""
abstract type ICAGDeriv end

"""
Derivative for ``(1/a_1)\\log\\cosh a_1 u``
"""
struct Tanh{T} <: ICAGDeriv
    a::T
end
function update!(f::Tanh{T}, U::AbstractMatrix{T}, E::AbstractVector{T}) where {T}
    n,k = size(U)
    a = f.a
    @inbounds for j in 1:k
        _s = zero(T)
        @fastmath for i in 1:n
            t = tanh(a * U[i,j])
            U[i,j] = t
            _s += a * (1 - t^2)
        end
        E[j] = _s / n
    end
end

"""
Derivative for ``-e^{\\frac{-u^2}{2}}``
"""
struct Gaus <: ICAGDeriv end
function update!(f::Gaus, U::AbstractMatrix{T}, E::AbstractVector{T}) where {T}
    n,k = size(U)
    @inbounds for j in 1:k
        _s = zero(T)
        for i in 1:n
            u = U[i,j]
            u2 = u^2
            e = exp(-u2/2)
            U[i,j] = u * e
            _s += (1 - u2) * e
        end
        E[j] = _s / n
    end
end


# Fast ICA

"""
    fastica!(W, X, fun, maxiter, tol, verbose)

Invoke the Fast ICA algorithm[^1].

**Parameters:**
- `W`: The initial un-mixing matrix, of size ``(m, k)``. The function updates this matrix inplace.
- `X`: The data matrix, of size ``(m, n)``. This matrix is input only, and won't be modified.
- `fun`: The approximate neg-entropy functor of type [`ICAGDeriv`](@ref).
- `maxiter`: Maximum number of iterations.
- `tol`: Tolerable change of `W` at convergence.

Returns the updated `W`.

**Note:** The number of components is inferred from `W` as `size(W, 2)`.
"""
function fastica!(W::DenseMatrix{T},         # initialized component matrix, size (m, k)
                  X::DenseMatrix{T},         # (whitened) observation sample matrix, size(m, n)
                  fun::ICAGDeriv,            # approximate neg-entropy functor
                  maxiter::Int,              # maximum number of iterations
                  tol::Real) where {T<:Real} # convergence tolerance

    # argument checking
    m = size(W, 1)
    k = size(W, 2)
    size(X, 1) == m || throw(DimensionMismatch("Sizes of W and X mismatch."))
    n = size(X, 2)
    k <= min(m, n) || throw(DimensionMismatch("k must not exceed min(m, n)."))

    @debug "FastICA Algorithm" m=m n=n k=k

    # pre-allocated storage
    Wp = similar(W)                # to store previous version of W
    U  = Matrix{T}(undef, n, k)    # to store w'x & g(w'x)
    Y  = Matrix{T}(undef, m, k)    # to store E{x g(w'x)} for components
    E1 = Vector{T}(undef, k)       # store E{g'(w'x)} for components

    # normalize each column
    for j = 1:k
        w = view(W,:,j)
        rmul!(w, 1.0 / sqrt(sum(abs2, w)))
    end

    # main loop
    chg = T(NaN)
    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1
        copyto!(Wp, W)

        # apply W of previous step
        mul!(U, transpose(X), W) # u <- w'x

        # compute g(w'x) --> U and E{g'(w'x)} --> E1
        update!(fun, U, E1)

        # compute E{x g(w'x)} --> Y
        rmul!(mul!(Y, X, U), 1 / n)

        # update w: E{x g(w'x)} - E{g'(w'x)}w, i.e. w := y - e1 * w
        for j = 1:k
            w = view(W,:,j)
            y = view(Y,:,j)
            e1 = E1[j]
            @. w = y - e1 * w
        end

        # symmetric decorrelation: W <- W * (W'W)^{-1/2}
        copyto!(W, W * _invsqrtm!(W'W))

        # compare with Wp to evaluate a conversion change
        chg = maximum(abs.(abs.(diag(W*Wp')) .- 1))
        converged = (chg < tol)

        @debug "Iteration $t" change=chg tolerance=tol
    end
    converged || throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))
    return W
end

#### interface function
"""
    fit(ICA, X, k; ...)

Perform ICA over the data set given in `X`.

**Parameters:**
-`X`: The data matrix, of size ``(m, n)``. Each row corresponds to a mixed signal,
while each column corresponds to an observation (*e.g* all signal value at a particular time step).
-`k`: The number of independent components to recover.

**Keyword Arguments:**
- `alg`: The choice of algorithm (*default* `:fastica`)
- `fun`: The approx neg-entropy functor (*default* [`Tanh`](@ref))
- `do_whiten`: Whether to perform pre-whitening (*default* `true`)
- `maxiter`: Maximum number of iterations (*default* `100`)
- `tol`: Tolerable change of ``W`` at convergence (*default* `1.0e-6`)
- `mean`: The mean vector, which can be either of:
    - `0`: the input data has already been centralized
    - `nothing`: this function will compute the mean (*default*)
    - a pre-computed mean vector
- `winit`: Initial guess of ``W``, which should be either of:
    - empty matrix: the function will perform random initialization (*default*)
    - a matrix of size ``(k, k)`` (when `do_whiten`)
    - a matrix of size ``(m, k)`` (when `!do_whiten`)

Returns the resultant ICA model, an instance of type [`ICA`](@ref).

**Note:** If `do_whiten` is `true`, the return `W` satisfies
``\\mathbf{W}^T \\mathbf{C} \\mathbf{W} = \\mathbf{I}``,
otherwise ``W`` is orthonormal, *i.e* ``\\mathbf{W}^T \\mathbf{W} = \\mathbf{I}``.
"""
function fit(::Type{ICA}, X::AbstractMatrix{T},# sample matrix, size (m, n)
             k::Int;                           # number of independent components
             alg::Symbol=:fastica,             # choice of algorithm
             fun::ICAGDeriv=Tanh(one(T)),      # approx neg-entropy functor
             do_whiten::Bool=true,             # whether to perform pre-whitening
             maxiter::Integer=100,             # maximum number of iterations
             tol::Real=1.0e-6,                 # convergence tolerance
             mean=nothing,                     # pre-computed mean
             winit::Matrix{T}=zeros(T,0,0)     # init guess of W, size (m, k)
            ) where {T<:Real}

    # check input arguments
    m, n = size(X)
    n > 1 || error("There must be more than one samples, i.e. n > 1.")
    k <= min(m, n) || error("k must not exceed min(m, n).")

    alg == :fastica || error("alg must be :fastica")
    maxiter > 1 || error("maxiter must be greater than 1.")
    tol > 0 || error("tol must be positive.")

    # preprocess data
    mv = preprocess_mean(X, mean)
    Z::Matrix{T} = centralize(X, mv)

    W0= zeros(T, 0, 0)  # whitening matrix
    if do_whiten
        C = rmul!(Z * transpose(Z), 1.0 / (n - 1))
        Efac = eigen(C)
        ord = sortperm(Efac.values; rev=true)
        (v, P) = extract_kv(Efac, ord, k)
        W0 = rmul!(P, Diagonal(1 ./ sqrt.(v)))
        Z = W0'Z
    end

    # initialize
    W = (isempty(winit) ? randn(T, size(Z,1), k) : copy(winit))

    # invoke core algorithm
    fastica!(W, Z, fun, maxiter, tol)

    # construct model
    if do_whiten
        W = W0 * W
    end
    return ICA(mv, W)
end

