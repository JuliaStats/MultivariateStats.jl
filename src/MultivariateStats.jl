module MultivariateStats
    using StatsBase

    import Base: length, show, dump
    import StatsBase: fit

    export 

    ## common
    centralize,         # subtract a mean vector from each column
    decentralize,       # add a mean vector to each column
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    projection,         # the projection matrix
    reconstruct,        # reconstruct the input (approximately) given the output
    transform,          # apply a model to transform a vector or a matrix

    ## pca
    PCA,                # Type: Principal Component Analysis model

    pca,                # Perform PCA over a given data set
    pcacov,             # PCA based on covariance
    pcastd,             # PCA based on singular value decomposition
    principalratio,     # the ratio of variances preserved in the principal subspace
    principalvar,       # the variance along a specific principal direction
    principalvars,      # the variances along all principal directions
    
    tprincipalvar,      # total principal variance, i.e. sum(principalvars(M))
    tresidualvar,       # total residual variance
    tvar,               # total variance

    ## cca
    CCA,                # Type: Correlation Component Analysis model

    xindim,             # input dimension of X
    yindim,             # input dimension of Y
    xmean,              # sample mean of X
    ymean,              # sample mean of Y
    xprojection,        # projection matrix for X
    yprojection,        # projection matrix for Y
    xtransform,         # transform for X
    ytransform,         # transform for Y
    correlations        # correlations of all projected directions

    ## source files
    include("common.jl")
    include("pca.jl")
    include("cca.jl")

end # module
