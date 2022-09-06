# Correspondence Analysis and Multiple Correspondence Analysis

# Needed for the plotting function below
#using Printf, PyPlot

#==
References:
https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
https://www.stata.com/manuals/mvmca.pdf
https://www.stata.com/manuals/mvca.pdf
https://en.wikipedia.org/wiki/Multiple_correspondence_analysis
https://pca4ds.github.io
https://maths.cnam.fr/IMG/pdf/ClassMCA_cle825cfc.pdf
==#

"""
Correspondence Analysis
"""
mutable struct CA{T<:Real} <: LinearDimensionalityReduction

    # The data matrix
    Z::Array{T}

    # The residuals
    R::Array{T}

    # Row and column masses (means)
    rm::Vector{T}
    cm::Vector{T}

    # The standardized residuals
    SR::Array{T}

    # Object scores
    F::Array{T}

    # Variable scores
    G::Array{T}

    # Inertia (eigenvalues of the indicator matrix)
    I::Vector{T}
end

# Constructor

function CA(X)

    # Convert to proportions
    X = X ./ sum(X)

    # Calculate row and column means
    r = sum(X, dims = 2)[:]
    c = sum(X, dims = 1)[:]

    # Center the data matrix to create residuals
    R = X - r * c'

    # Standardize the data matrix to create standardized residuals
    Wr = Diagonal(sqrt.(r))
    Wc = Diagonal(sqrt.(c))
    SR = Wr \ R / Wc

    T = eltype(X)
    return CA(X, R, r, c, SR, zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0))
end

function fit!(ca::CA, d::Int)

    # Get the object factor scores (F) and variable factor scores (G).
    P, D, Q = svd(ca.SR)
    Dq = Diagonal(D)[:, 1:d]

    # Check that there are no repeated non-zero eigenvalues.
    d = diff(D[D .> 1e-10])
    if maximum(d) >= -1e-10
        @warn("The indicator matrix has repeated non-zero eigenvalues")
    end

    Wr = Diagonal(sqrt.(ca.rm))
    Wc = Diagonal(sqrt.(ca.cm))
    ca.F = Wr \ P * Dq
    ca.G = Wc \ Q * Dq

    # Get the eigenvalues
    ca.I = D .^ 2
end

function fit(::Type{CA}; X::AbstractMatrix, d::Int = 5)
    ca = CA(X)
    fit!(ca, d)
    return ca::CA
end

"""
    ca

Fit a correspondence analysis using the data array `X` whose rows are
the objects and columns are the variables.  The first `d` components
are retained.
"""
function ca(X, d)
    c = fit(CA, X, d)
    return c
end

objectscores(ca::CA) = ca.F
variablescores(ca::CA) = ca.G
inertia(ca::CA) = ca.I

"""
Multiple Correspondence Analysis
"""
mutable struct MCA{T<:Real} <: LinearDimensionalityReduction

    # The underlying corresponence analysis
    C::CA{T}

    # Variable names
    vnames::Vector{String}

    # Map values to integer positions
    rd::Vector{Dict}

    # Map integer positions to values
    dr::Vector{Dict}

    # Split the variable scores into separate arrays for
    # each variable.
    Gv::Vector{Matrix{T}}

    # Number of nominal variables
    K::Int

    # Total number of categories in all variables
    J::Int

    # Eigenvalues
    unadjusted_eigs::Vector{Float64}
    benzecri_eigs::Vector{Float64}
    greenacre_eigs::Vector{Float64}
end

objectscores(mca::MCA) = mca.C.F
variablescores(mca::MCA) = mca.Gv

# Split the variable scores to a separate array for each
# variable.
function xsplit(G, rd)
    K = [length(di) for di in rd]
    Js = cumsum(K)
    Js = vcat(1, 1 .+ Js)
    Gv = Vector{Matrix{eltype(G)}}()
    for j in eachindex(K)
        g = G[Js[j]:Js[j+1]-1, :]
        push!(Gv, g)
    end
    return Gv
end

# Calculate the eigenvalues with different debiasings.
function get_eigs(I, K, J)
    ben = zeros(length(I))
    gra = zeros(length(I))
    Ki = 1 / K
    f = K / (K - 1)
    for i in eachindex(I)
        if I[i] > Ki
            ben[i] = (f * (I[i] - Ki))^2
        end
    end

    unadjusted = I ./ sum(I)
    gt = f * (sum(abs2, I) - (J - K) / K^2)

    return unadjusted, ben ./ sum(ben), ben ./ gt
end

# constructor

function MCA(Z; vnames = [])

    if length(vnames) == 0
        vnames = ["v$(j)" for j = 1:size(Z, 2)]
    end

    # Get the indicator matrix
    X, rd, dr = make_indicators(Z)

    # Create the underlying correspondence analysis value
    C = CA(X)

    # Number of nominal variables
    K = size(Z, 2)

    # Total number of categories in all variables
    J = size(X, 2)

    return MCA(C, vnames, rd, dr, Matrix{Float64}[], K, J, zeros(0), zeros(0), zeros(0))
end

"""
    mca

Fit a multiple correspondence analysis using the columns of `Z` as the
variables.  The first `d` components are retained.  If `Z` is a
dataframe then the column names are used as variable names, otherwise
variable names may be provided as `vnames`.
"""
function mca(Z, d::Int; vnames = [])
    m = MCA(Z; vnames)
    fit!(m, d)
    return m
end

function fit(::Type{MCA}, Z::AbstractMatrix, d::Int; vnames = [])
    return mca(Z, d; vnames)
end

function fit!(mca::MCA, d::Int)

    fit!(mca.C, d)

    # Split the variable scores into separate arrays for each variable.
    mca.Gv = xsplit(mca.C.G, mca.rd)

    una, ben, gra = get_eigs(mca.C.I, mca.J, mca.K)

    mca.unadjusted_eigs = una
    mca.benzecri_eigs = ben
    mca.greenacre_eigs = gra

    return mca
end

# Create an indicator matrix corresponding to the distinct
# values in z.  Also returns dictionaries mapping the unique
# values to column offsets, and mapping the column offsets
# to the unique values.
function make_single_indicator(z)

    n = length(z)

    # Unique values of the variable
    uq = sort(unique(z))

    if length(uq) > 50
        @warn("Nominal variable has more than 50 levels")
    end

    # Recoding dictionary, maps each distinct value in z to
    # an offset
    rd = Dict{eltype(z),Int}()
    for (j, v) in enumerate(uq)
        if !ismissing(v)
            rd[v] = j
        end
    end

    # Number of unique values of the variable excluding missing
    m = length(rd)

    # The indicator matrix
    X = zeros(n, m)
    for (i, v) in enumerate(z)
        if ismissing(v)
            # Missing values are treated as uniform across the levels.
            X[i, :] .= 1 / m
        else
            X[i, rd[v]] = 1
        end
    end

    # Reverse the recoding dictionary
    rdi = Dict{Int,eltype(z)}()
    for (k, v) in rd
        rdi[v] = k
    end

    return X, rd, rdi
end

# Create an indicator matrix for the nominal data matrix Z.
# In addition to the indicator matrix, return vectors of
# dictionaries mapping levels to positions and positions
# to levels for each variable.
function make_indicators(Z)

    rd, rdr = Dict[], Dict[]
    XX = []
    for j = 1:size(Z, 2)
        X, di, dir = make_single_indicator(Z[:, j])
        push!(rd, di)
        push!(rdr, dir)
        push!(XX, X)
    end
    I = hcat(XX...)

    return I, rd, rdr
end

# Return a table summarizing the inertia.
function inertia(mca::MCA)
    inr = (
        Raw = mca.C.I,
        Unadjusted = mca.unadjusted_eigs,
        Benzecri = mca.benzecri_eigs,
        Greenacre = mca.greenacre_eigs,
    )
    return inr
end

# Plot the category scores for components numbered 'x' and 'y'.  Ordered factors
# are connected with line segments.
#function variable_plot(mca::MCA; x = 1, y = 2, vnames = [], ordered = [], kwargs...)

#    fig = PyPlot.figure()
#    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#    ax.grid(true)

# Set up the colormap
#    cm = get(kwargs, :cmap, PyPlot.get_cmap("tab10"))

# Set up the axis limits
#    mn = 1.2 * minimum(mca.C.G, dims = 1)
#    mx = 1.2 * maximum(mca.C.G, dims = 1)
#    xlim = get(kwargs, :xlim, [mn[x], mx[x]])
#    ylim = get(kwargs, :ylim, [mn[y], mx[y]])
#    ax.set_xlim(xlim...)
#    ax.set_ylim(ylim...)

#    for (j, g) in enumerate(mca.Gv)

#        if mca.vnames[j] in ordered
#            PyPlot.plot(g[:, x], g[:, y], "-", color = cm(j))
#        end

#        dr = mca.dr[j]
#        vn = length(vnames) > 0 ? vnames[j] : ""
#        for (k, v) in dr
#            if vn != ""
#                lb = "$(vn)-$(v)"
#            else
#                lb = v
#            end
#            ax.text(g[k, x], g[k, y], lb, color = cm(j), ha = "center", va = "center")
#        end
#    end

#    inr = inertia(mca)
#    PyPlot.xlabel(@sprintf("Dimension %d (%.2f%%)", x, 100 * inr[x, :Greenacre]))
#    PyPlot.ylabel(@sprintf("Dimension %d (%.2f%%)", y, 100 * inr[y, :Greenacre]))

#    return fig
#end
