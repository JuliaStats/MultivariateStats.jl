# Independent Component Analysis

#### FastICA type

mutable struct ICA{T<:Real}
    mean::Vector{T}   # mean vector, of length m (or empty to indicate zero mean)
    W::Matrix{T}      # component coefficient matrix, of size (m, k)
end

indim(M::ICA) = size(M.W, 1)
outdim(M::ICA) = size(M.W, 2)
Base.mean(M::ICA) = fullmean(indim(M), M.mean)

transform(M::ICA, x::AbstractVecOrMat) = At_mul_B(M.W, centralize(x, M.mean))


#### core algorithm

# the abstract type for all g functions:
#
# Let f be an instance of such type, then
#
#   evaluate(f, x) --> (v, d)
#
# It returns a function value v, and derivative d
#
abstract type ICAGDeriv{T<:Real} end

struct Tanh{T} <: ICAGDeriv{T}
    a::T
end

evaluate(f::Tanh{T}, x::T) where {T<:Real} = (a = f.a; t = tanh(a * x); (t, a * (1 - t * t)))

struct Gaus{T} <: ICAGDeriv{T} end
evaluate(f::Gaus{T}, x::T) where {T<:Real} = (x2 = x * x; e = exp(-x2/2); (x * e, (1 - x2) * e))

## a function to get a g-fun

icagfun(fname::Symbol, ::Type{T} = Float64) where {T<:Real}=
    fname == :tanh ? Tanh{T}(1.0) :
    fname == :gaus ? Gaus{T}() :
    error("Unknown gfun $(fname)")

icagfun(fname::Symbol, a::T) where {T<:Real} =
    fname == :tanh ? Tanh(a) :
    fname == :gaus ? error("The gfun $(fname) has no parameters") :
    error("Unknown gfun $(fname)")

# Fast ICA
#
# Reference:
#
#   Aapo Hyvarinen and Erkki Oja
#   Independent Component Analysis: Algorithms and Applications.
#   Neural Network 13(4-5), 2000.
#
function fastica!(W::DenseMatrix{T},      # initialized component matrix, size (m, k)
                  X::DenseMatrix{T},      # (whitened) observation sample matrix, size(m, n)
                  fun::ICAGDeriv{T},      # approximate neg-entropy functor
                  maxiter::Int,           # maximum number of iterations
                  tol::Real,              # convergence tolerance
                  verbose::Bool) where {T<:Real}          # whether to show iterative info

    # argument checking
    m = size(W, 1)
    k = size(W, 2)
    size(X, 1) == m || throw(DimensionMismatch("Sizes of W and X mismatch."))
    n = size(X, 2)
    k <= min(m, n) || throw(DimensionMismatch("k must not exceed min(m, n)."))

    if verbose
        @printf("FastICA Algorithm (m = %d, n = %d, k = %d)\n", m, n, k)
        println("============================================")
    end

    # pre-allocated storage
    Wp = Matrix{T}(m, k)    # to store previous version of W
    U  = Matrix{T}(n, k)    # to store w'x & g(w'x)
    Y  = Matrix{T}(m, k)    # to store E{x g(w'x)} for components
    E1 = Vector{T}(k)       # store E{g'(w'x)} for components

    # normalize each column
    for j = 1:k
        w = view(W,:,j)
        scale!(w, 1.0 / sqrt(sum(abs2, w)))
    end

    # main loop
    chg = NaN
    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1
        copy!(Wp, W)

        # apply W of previous step
        At_mul_B!(U, X, W)  # u <- w'x

        # compute g(w'x) --> U and E{g'(w'x)} --> E1
        _s = 0.0
        for j = 1:k
            for i = 1:n
                u, v = evaluate(fun, U[i,j])
                U[i,j] = u
                _s += v
            end
            E1[j] = _s / n
        end

        # compute E{x g(w'x)} --> Y
        scale!(A_mul_B!(Y, X, U), 1.0 / n)

        # update w: E{x g(w'x)} - E{g'(w'x)} w := y - e1 * w
        for j = 1:k
            w = view(W,:,j)
            y = view(Y,:,j)
            e1 = E1[j]
            for i = 1:m
                w[i] = y[i] - e1 * w[i]
            end
        end

        # symmetric decorrelation: W <- W * (W'W)^{-1/2}
        copy!(W, W * _invsqrtm!(W'W))

        # compare with Wp
        chg = 0.0
        for j = 1:k
            s = 0.0
            w = view(W,:,j)
            wp = view(Wp,:,j)
            s = abs(abs(dot(w, wp))-1)
            if s > chg 
                chg = s
            end
        end
        converged = (chg < tol)

        if verbose
            @printf("Iter %4d:  change = %.6e\n", t, chg)
        end
    end
    converged || throw(ConvergenceException(maxiter, chg, oftype(chg, tol)))
    return W
end

#### interface function

function fit(::Type{ICA}, X::DenseMatrix{T},                # sample matrix, size (m, n)
                          k::Int;                           # number of independent components
                          alg::Symbol=:fastica,             # choice of algorithm
                          fun::ICAGDeriv=icagfun(:tanh, T), # approx neg-entropy functor
                          do_whiten::Bool=true,             # whether to perform pre-whitening
                          maxiter::Integer=100,             # maximum number of iterations
                          tol::Real=1.0e-6,                 # convergence tolerance
                          mean=nothing,                     # pre-computed mean
                          winit::Matrix{T}=zeros(T,0,0),    # init guess of W, size (m, k)
                          verbose::Bool=false) where {T<:Real}             # whether to display iterations

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

    W0= zeros(T, 0,0)    # whitening matrix
    if do_whiten
        C = scale!(A_mul_Bt(Z, Z), 1.0 / (n - 1))
        Efac = eigfact(C)
        ord = sortperm(Efac.values; rev=true)
        (v, P) = extract_kv(Efac, ord, k)
        W0 = scale!(P, 1 ./ sqrt.(v))
        # println(W0' * C * W0)
        Z = W0'Z
    end

    # initialize
    W = (isempty(winit) ? randn(T, size(Z,1), k) : copy(winit))

    # invoke core algorithm
    fastica!(W, Z, fun, maxiter, tol, verbose)

    # construct model
    if do_whiten
        W = W0 * W
    end
    return ICA(mv, W)
end
