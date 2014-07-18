module MultivariateStats
    using StatsBase
    using ArrayViews

    import Base: length, show, dump, evaluate
    import StatsBase: fit, predict

    export 

    ## common
    evaluate,           # evaluate discriminant function values (imported from Base)
    predict,            # use a model to predict responses (imported from StatsBase)
    fit,                # fit a model to data (imported from StatsBase)
    centralize,         # subtract a mean vector from each column
    decentralize,       # add a mean vector to each column
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    projection,         # the projection matrix
    reconstruct,        # reconstruct the input (approximately) given the output
    transform,          # apply a model to transform a vector or a matrix

    ## pca
    PCA,                # Type: Principal Component Analysis model

    pcacov,             # PCA based on covariance
    pcasvd,             # PCA based on singular value decomposition of input data
    principalratio,     # the ratio of variances preserved in the principal subspace
    principalvar,       # the variance along a specific principal direction
    principalvars,      # the variances along all principal directions
    
    tprincipalvar,      # total principal variance, i.e. sum(principalvars(M))
    tresidualvar,       # total residual variance
    tvar,               # total variance

    ## cca
    CCA,                # Type: Correlation Component Analysis model

    ccacov,             # CCA based on covariances
    ccasvd,             # CCA based on singular value decomposition of input data

    xindim,             # input dimension of X
    yindim,             # input dimension of Y
    xmean,              # sample mean of X
    ymean,              # sample mean of Y
    xprojection,        # projection matrix for X
    yprojection,        # projection matrix for Y
    xtransform,         # transform for X
    ytransform,         # transform for Y
    correlations,       # correlations of all projected directions

    ## cmds
    classical_mds,      # perform classical MDS over a given distance matrix

    gram2dmat, gram2dmat!,  # Gram matrix => Distance matrix
    dmat2gram, dmat2gram!,  # Distance matrix => Gram matrix

    ## lda
    Discriminant,           # Abstract Type: for all discriminant functionals
    LinearDiscriminant,     # Type: Linear Discriminant functional 
    MulticlassLDAStats,     # Type: Statistics required for training multi-class LDA
    MulticlassLDA,          # Type: Multi-class LDA model

    ldacov,                 # Linear discriminant analysis based on covariances

    classweights,           # class-specific weights
    classmeans,             # class-specific means
    withclass_scatter,      # with-class scatter matrix
    betweenclass_scatter,   # between-class scatter matrix
    multiclass_lda_stats,   # compute statistics for multiclass LDA training
    multiclass_lda,         # train multi-class LDA based on statistics
    mclda_solve,            # solve multi-class LDA projection given scatter matrices
    mclda_solve!,           # solve multi-class LDA projection (inputs are overriden)
    whitening, whitening!   # compute whitening transform

    ## source files
    include("common.jl")
    include("pca.jl")
    include("cca.jl")
    include("cmds.jl")
    include("lda.jl")

end # module
