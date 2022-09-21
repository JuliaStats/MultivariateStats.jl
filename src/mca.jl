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
struct CA{T<:Real} <: LinearDimensionalityReduction

    # The data matrix
    X::Array{T}

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

"""
    fit(CA, X; ...)

Peform a Correspondence Analysis using the data in `X`.

**Keyword Arguments**

- `d` the number of dimensions to retain.

**Notes:**

- The matrix `X` should contain numerical data for which it makes
  sense to use chi^2 distance to compare rows.  Most commonly this
  is an indicator matrix in which each row contains a single 1 with
  the remaining values all being 0.
- See `MCA` for a more general analysis that takes a dataframe
  with multiple nominal columns as input and performs a CA on
  its indicator matrix.
"""
function fit(::Type{CA}, X::AbstractMatrix{T}; d::Int = 5) where{T}

    # Convert to proportions
    X = X ./ sum(X)

    # Calculate row and column margins
    rm = sum(X, dims = 2)[:]
    cm = sum(X, dims = 1)[:]

    # Center the data matrix to create residuals
    R = X - rm * cm'

    # Standardize the data matrix to create standardized residuals
    Wr = Diagonal(sqrt.(rm))
    Wc = Diagonal(sqrt.(cm))
    SR = Wr \ R / Wc

    # Get the object scores (F) and variable scores (G).
    P, D, Q = svd(SR)
    Dq = Diagonal(D)[:, 1:d]

    # Check that there are no repeated non-zero eigenvalues.
    d = diff(D[D .> 1e-10])
    if maximum(d) >= -1e-10
        @warn("The indicator matrix has repeated non-zero eigenvalues")
    end

    Wr = Diagonal(sqrt.(rm))
    Wc = Diagonal(sqrt.(cm))
    F = Wr \ P * Dq
    G = Wc \ Q * Dq

    # Get the eigenvalues
    I = D .^ 2

    return CA(X, R, rm, cm, SR, F, G, I)
end

objectscores(ca::CA) = ca.F
variablescores(ca::CA) = ca.G
inertia(ca::CA) = ca.I

"""
Multiple Correspondence Analysis
"""
struct MCA{T<:Real} <: LinearDimensionalityReduction

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

"""
    fit(MCA, X; ...)

Fit a multiple correspondence analysis using the columns of `X` as the variables.

**Keyword Arguments**

- `d`: The number of dimensions to retain.
- `vnames`: Variable names, if `X` is a data frame then the column names are the
  default variable names but will be replaced with these values if provided.

**Notes:**

- Missing values are recoded as fractional indicators, i.e. if there are k distinct
  levels of a variable it is coded as 1/k, 1/k, ..., 1/k.
"""
function fit(::Type{MCA}, X; d::Int=5, vnames=[])

    if length(vnames) == 0 && typeof(X) <: AbstractDataFrame
        vnames = names(X)
    elseif length(vnames) == 0
        vnames = ["v$(j)" for j = 1:size(Z, 2)]
    end

    # Get the indicator matrix
    XI, rd, dr = make_indicators(X)

    # Create the underlying correspondence analysis value
    C = fit(CA, XI; d=d)

    # Number of nominal variables
    K = size(X, 2)

    # Total number of categories in all variables
    J = size(XI, 2)

    # Split the variable scores into separate arrays for each variable.
    Gv = xsplit(C.G, rd)

    una, ben, gra = get_eigs(C.I, J, K)

    return MCA(C, vnames, rd, dr, Gv, K, J, una, ben, gra)
end

# Create an indicator matrix corresponding to the distinct
# values in the vector 'z'.  Also returns dictionaries mapping
# the unique values to column offsets, and mapping the column
# offsets to the unique values.
function make_single_indicator(z::Vector{T}) where{T}

    n = length(z)

    # Unique values of the variable
    uq = sort(unique(z))

    if length(uq) > 50
        @warn("Nominal variable has more than 50 levels")
    end

    # Recoding dictionary, maps each distinct value in z to
    # an offset
    rd = Dict{T,Int}()
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
    rdi = Dict{Int,T}()
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
