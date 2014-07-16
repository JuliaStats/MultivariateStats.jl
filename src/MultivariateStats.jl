module MultivariateStats
    using StatsBase

    import Base: length, show, dump
    import StatsBase: fit

    export 

    # types
    PCA,        # Principal Component Analysis model

    # functions
    centralize,         # subtract a mean vector from each column
    decentralize,       # add a mean vector to each column
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    pcacov,             # PCA based on covariance
    pcastd,             # PCA based on singular value decomposition
    principalratio,     # the ratio of variances preserved in the principal subspace
    principalvar,       # the variance along a specific principal direction
    principalvars,      # the variances along all principal directions
    projection,         # the projection matrix
    tprincipalvar,      # total principal variance, i.e. sum(principalvars(M))
    tresidualvar,       # total residual variance
    tvar,               # total variance
    reconstruct,        # reconstruct the input (approximately) given the output
    transform           # apply a model to transform a vector or a matrix

    ## source files
    include("common.jl")
    include("pca.jl")

end # module
