# Correspondence Analysis and Multiple Correspondence Analysis

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
    X::Matrix{T}

    # The residuals
    R::Matrix{T}

    # Row masses
    rm::Vector{T}

    # Column masses
    cm::Vector{T}

    # The standardized residuals
    SR::Matrix{T}

    # Object coordinates in standard coordinates
    FS::Matrix{T}

    # Variable coordinates in standard coordinates
    GS::Matrix{T}

    # Inertia (eigenvalues of the indicator matrix)
    I::Vector{T}

    # Standard normalization or principal normalization
    normalize::String

    # Use either the Burt method or the indicator method
    method::String
end

function show(io::IO, ca::CA)
    nobs, nvar = size(ca.X)
    print(
        io,
        "CA(nobs = $nobs, nvar = $nvar, method = $(ca.method), normalize = $(ca.normalize))",
    )
end

function print_inertia(io, I)
    xx = hcat(I, I, I)
    xx[:, 2] = xx[:, 1] ./ sum(xx[:, 1])
    xx[:, 3] = cumsum(xx[:, 2])
    ii = findfirst(xx[:, 3] .>= 0.999)
    xx = xx[1:ii, :]
    vn = ["Inertia", "Prop inertia", "Cumulative inertia"]
    cft = CoefTable(xx, vn, string.("", 1:ii))
    println(io, cft)
end

function build_coeftable(ca, vc, stats, rn)
    cn = ["Mass"]
    xx = [ca.cm]
    d = size(ca.GS, 2)
    for j = 1:d
        push!(cn, "Coord-$(j)")
        push!(xx, vc[:, j])
        push!(cn, "SqCorr-$(j)")
        push!(xx, stats.sqcorr_col[:, j])
        push!(cn, "RelContrib-$(j)")
        push!(xx, stats.relcontrib_col[:, j])
    end
    xx = hcat(xx...)
    cft = CoefTable(xx, cn, rn)
    return cft
end

function show(io::IO, ::MIME"text/plain", ca::CA)
    nobs, nvar = size(ca.X)
    stats = ca_stats(ca)

    println(
        io,
        "CA(nobs = $nobs, nvar = $nvar, method = $(ca.method), normalize = $(ca.normalize))",
    )
    print_inertia(io, ca.I)
    vc = variable_coords(ca)
    m, d = size(ca.GS)
    println(io, "\nVariable coordinates:")
    rn = string.("", 1:m)
    cft = build_coeftable(ca, vc, stats, rn)
    print(io, cft)
    print(io, "\n\n")
end

# Squared correlation of each variable with each component.
function sqcorr(ca::CA, axis)
    if axis == :row
        z = ca.rm ./ sum(abs2, ca.SR; dims = 2)[:]
        p = length(ca.rm)
        d = size(ca.GS, 2)
        if ca.method == "burt"
            # Not clear how to compute this so report zero for now
            return zeros(p, d)
        end
        f = z * ones(d)'
        return object_coords(ca; normalize = "principal") .^ 2 .* f
    elseif axis == :col
        z = ca.cm ./ sum(abs2, ca.SR; dims = 1)[:]
        p = length(ca.cm)
        d = size(ca.GS, 2)
        f = z * ones(d)'
        if ca.method == "burt"
            # Not clear how to compute this so report zero for now
            return zeros(p, d)
        end
        return variable_coords(ca; normalize = "principal") .^ 2 .* f
    else
        error("Unknown axis '$(axis)'")
    end
end

# Relative contributions of each variable to each component.
function relcontrib(ca::CA, axis)
    e = ca.method == "indicator" ? 0.5 : 1
    if axis == :row
        d = size(ca.GS, 2)
        z = ca.rm * (ca.I[1:d] .^ -e)'
        return object_coords(ca; normalize = "principal") .^ 2 .* z
    elseif axis == :col
        d = size(ca.GS, 2)
        z = ca.cm * (ca.I[1:d] .^ -e)'
        return variable_coords(ca; normalize = "principal") .^ 2 .* z
    else
        error("Unknown axis '$(axis)'")
    end
end

# Return several fit statistics.
function ca_stats(ca::CA)
    cr = sqcorr(ca, :col)
    rc = relcontrib(ca, :col)
    return (sqcorr_col = cr, relcontrib_col = rc)
end

# Multiply corresponding columns of Q by -1 as needed
# to make the first non-zero value in each column of Q
# positive.  This is the method used by Stata to identify
# the coefficients.
function orient(P, Q, e = 1e-12)
    for j = 1:size(Q, 2)
        i = findfirst(abs.(Q[:, j]) .>= e)
        if Q[i, j] < 0
            Q[:, j] .*= -1
            P[:, j] .*= -1
        end
    end
    return P, Q
end

function mca_check_args(normalize, method)
    if !(normalize in ["standard", "principal"])
        error("normalize = '$normalize' should be 'standard' or 'principal'")
    end
    if !(method in ["indicator", "burt"])
        error("method = '$method' should be 'indicator' or 'burt'")
    end
end

"""
    fit(CA, X; ...)

Peform a Correspondence Analysis using the data in `X`.

**Keyword Arguments**

- `d` the number of dimensions to retain.
- `normalize` a coordinate normalization method, either 'standard' or 'principal'.

**Notes:**

- The matrix `X` should contain numerical data for which it makes
  sense to use chi^2 distance to compare rows.  Most commonly this
  is an indicator matrix in which each row contains a single 1 with
  the remaining values all being 0.
- See `MCA` for a more general analysis that takes a dataframe
  with multiple nominal columns as input and performs a CA on
  its indicator matrix.
"""
function fit(
    ::Type{CA},
    X::AbstractMatrix{T};
    d::Int = 5,
    normalize = "standard",
    method = "indicator",
) where {T}

    mca_check_args(normalize, method)

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

    # Factor the standardized residual matrix
    P, D, Q = svd(SR)

    # Check that there are no repeated non-zero eigenvalues.
    di = diff(D[D.>1e-10])
    if maximum(di) >= -1e-10
        @warn("The indicator matrix has repeated non-zero eigenvalues")
    end

    # Get the eigenvalues
    I = method == "burt" ? D .^ 4 : D .^ 2

    # Reduce to the selected dimension
    P = P[:, 1:d]
    D = D[1:d]
    Q = Q[:, 1:d]

    # Flip the loading vectors to a standard orientation.
    P, Q = orient(P, Q)

    # Get the object scores (F) and variable scores (G).  These are the
    # standard coordinates.
    FS = Wr \ P
    GS = Wc \ Q

    ca = CA(X, R, rm, cm, SR, FS, GS, I, normalize, method)
    return ca
end

# Calculate the standard coordinates of any qualitative passive variables.
function quali_passive(ca::CA, passive; normalize = "principal")

    if size(passive, 1) == 0
        return
    end

    if length(size(passive)) != 2
        error("passive variable array must be two-dimensional")
    end

    (; X, GS) = ca
    PX = Matrix(passive)

    if size(PX, 1) != size(X, 1)
        @error("Passive data must have same leading axis length as active data.")
    end

    M = hcat(X, PX)
    B = M' * M
    B ./= sum(B)
    p = size(X, 2)
    B = B[p+1:end, 1:p]

    PGS = B * GS
    for k = 1:size(B, 1)
        PGS[k, :] ./= sum(B[k, :])
    end
    d = size(PGS, 2)
    if ca.method == "burt"
        PGS = PGS * Diagonal(1 ./ sqrt.(ca.I[1:d]))
    elseif ca.method == "indicator"
        PGS = PGS * Diagonal(1 ./ ca.I[1:d])
    else
        error("Unknown method '$(ca.method)'")
    end

    if normalize == "standard"
        return PGS
    elseif normalize == "principal"
        return PGS * Diagonal(sqrt.(ca.I[1:d]))
    else
        error("Unknown normalization '$(normalize)'")
    end

    return PGS
end

function object_coords(ca::CA; normalize = ca.normalize)
    if normalize == "standard"
        ca.FS
    elseif normalize == "principal"
        d = size(ca.FS, 2)
        return ca.FS * Diagonal(sqrt.(ca.I[1:d]))
    else
        error("Unknown normalization '$(normalize)'")
    end
end

inertia(ca::CA) = ca.I

function variable_coords(ca::CA; normalize = ca.normalize)
    (; GS) = ca

    d = size(GS, 2)
    if normalize == "standard"
        return GS
    elseif normalize == "principal"
        return GS * Diagonal(sqrt.(ca.I[1:d]))
    else
        error("Unknown normalization '$(normalize)'")
    end
end

"""
Multiple Correspondence Analysis
"""
mutable struct MCA{T<:Real} <: LinearDimensionalityReduction

    # The underlying corresponence analysis
    C::CA{T}

    # Indicator matrix
    Inds::Matrix{Float64}

    # Variable names
    vnames::Vector{String}

    # Map values to integer positions
    rd::Vector{Dict}

    # Map integer positions to values
    dr::Vector{Vector}

    # Number of nominal variables
    K::Int

    # Total number of categories in all variables
    J::Int

    # Eigenvalues
    unadjusted_eigs::Vector{Float64}
    benzecri_eigs::Vector{Float64}
    greenacre_eigs::Vector{Float64}
end

function expand_names(vnames, dr)
    names, levels = [], []
    for (j, v) in enumerate(vnames)
        u = dr[j]
        for k in eachindex(u)
            push!(names, v)
            push!(levels, u[k])
        end
    end
    return (Variable = names, Level = levels)
end

object_coords(mca::MCA; normalize = "principal") =
    object_coords(mca.C, normalize = normalize)

function variable_coords(mca::MCA; normalize = "principal")
    (; C, vnames, dr) = mca
    na = expand_names(vnames, dr)
    G = variable_coords(C, normalize = normalize)
    return (Variable = na.Variable, Level = na.Level, Coord = G)
end

function show(io::IO, mca::MCA)
    nobs, ninds = size(mca.C.X)
    nvar = length(mca.vnames)
    print(io, "MCA(nobs = $nobs, nvar = $nvar, ninds = $ninds)")
end

function show(io::IO, ::MIME"text/plain", mca::MCA)
    nobs, ninds = size(mca.C.X)
    stats = ca_stats(mca.C)
    nvar = length(mca.vnames)
    println(io, "MCA(nobs = $nobs, nvar = $nvar, ninds = $ninds)")
    print_inertia(io, mca.C.I)
    vc = variable_coords(mca.C; normalize = "principal")
    d = size(vc, 2)
    en = expand_names(mca.vnames, mca.dr)
    rn = ["$(a) $(b)" for (a, b) in zip(en.Variable, en.Level)]
    cft = build_coeftable(mca.C, vc, stats, rn)
    println(io, "\nVariable coordinates (principal normalization):")
    print(io, cft)
    print(io, "\n\n")
end

function ca_stats(mca::MCA)
    return ca_stats(mca.C)
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
function fit(
    ::Type{MCA},
    X;
    d::Int = 5,
    normalize = "standard",
    method = "indicator",
    vnames = [],
)

    if length(vnames) == 0 && typeof(X) <: AbstractDataFrame
        vnames = names(X)
    elseif length(vnames) == 0
        vnames = ["v$j" for j = 1:size(X, 2)]
    end

    # Get the indicator matrix
    XI, rd, dr = make_indicators(X)

    # Create the underlying correspondence analysis value
    C = fit(CA, XI; d = d, normalize = normalize, method = method)

    # Number of nominal variables
    K = size(X, 2)

    # Total number of categories in all variables
    J = size(XI, 2)

    una, ben, gra = get_eigs(C.I, J, K)

    mca = MCA(C, XI, vnames, rd, dr, K, J, una, ben, gra)

    return mca
end

function quali_passive(mca::MCA, passive; normalize = "principal")
    (; C) = mca
    if size(passive, 1) != size(C.X, 1)
        error("Wrong number of rows in passive data array")
    end

    PI, _, drp = make_indicators(passive)
    r = quali_passive(C, PI; normalize = normalize)

    vnames = if typeof(passive) <: AbstractDataFrame
        names(passive)
    else
        m = length(drp)
        string.("p", 1:m)
    end

    v = expand_names(vnames, drp)
    return (Variable = v.Variable, Level = v.Level, Coord = r)
end

# Create an indicator matrix corresponding to the distinct
# values in the vector 'z'.  Also returns dictionaries mapping
# the unique values to column offsets, and mapping the column
# offsets to the unique values.
function make_single_indicator(z::Vector{T}) where {T}

    n = length(z)

    # Unique values of the variable
    uq = sort(unique(z))

    if length(uq) > 50
        @warn("Nominal variable has more than 50 levels")
    end

    # Recoding dictionary, maps each distinct value in z to
    # an offset
    rd = Dict{T,Int}()
    rdi = []
    for (j, v) in enumerate(uq)
        if !ismissing(v)
            rd[v] = j
            push!(rdi, v)
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

    return X, rd, rdi
end

# Create an indicator matrix for the nominal data matrix Z.
# In addition to the indicator matrix, return vectors of
# dictionaries mapping levels to positions and positions
# to levels for each variable.
function make_indicators(Z)

    if size(Z, 1) == 0
        return zeros(0, 0), Dict[], Vector[]
    end

    rd, rdi = Dict[], Vector[]
    XX = []
    for j = 1:size(Z, 2)
        X, dv, di = make_single_indicator(Z[:, j])
        push!(rd, dv)
        push!(rdi, di)
        push!(XX, X)
    end
    I = hcat(XX...)

    return I, rd, rdi
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
