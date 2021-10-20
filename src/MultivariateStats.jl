module MultivariateStats
    using LinearAlgebra
    using StatsBase: SimpleCovariance, CovarianceEstimator, RegressionModel,
                     AbstractDataTransform, pairwise!
    import Statistics: mean, var, cov, covm
    import Base: length, size, show, dump
    import StatsBase: fit, predict, predict!, ConvergenceException, coef, weights,
                      dof, pairwise, r2
    import SparseArrays
    import LinearAlgebra: eigvals

    export

    ## common
    evaluate,           # evaluate discriminant function values
    predict,            # use a model to predict responses (imported from StatsBase)
    fit,                # fit a model to data (imported from StatsBase)
    centralize,         # subtract a mean vector from each column
    decentralize,       # add a mean vector to each column
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    projection,         # the projection matrix
    reconstruct,        # reconstruct the input (approximately) given the output
    transform,          # apply a model to transform a vector or a matrix

    # lreg
    llsq,               # Linear Least Square regression
    ridge,              # Ridge regression

    # whiten
    Whitening,          # Type: Whitening transformation

    invsqrtm,           # Compute inverse of matrix square root, i.e. inv(sqrtm(A))
    cov_whitening,      # Compute a whitening transform based on covariance
    cov_whitening!,     # Compute a whitening transform based on covariance (input will be overwritten)
    invsqrtm,           # Compute C^{-1/2}, i.e. inv(sqrtm(C))

    ## pca
    PCA,                # Type: Principal Component Analysis model

    pcacov,             # PCA based on covariance
    pcasvd,             # PCA based on singular value decomposition of input data
    principalratio,     # the ratio of variances preserved in the principal subspace
    principalvar,       # the variance along a specific principal direction
    principalvars,      # the variances along all principal directions

    tprincipalvar,      # total principal variance, i.e. sum(principalvars(M))
    tresidualvar,       # total residual variance
    loadings,           # model loadings

    ## ppca
    PPCA,               # Type: the Probabilistic PCA model

    ppcaml,             # Maximum likelihood probabilistic PCA
    ppcaem,             # EM algorithm for probabilistic PCA
    bayespca,           # Bayesian PCA
    loadings,           # factor loadings matrix

    ## kpca
    KernelPCA,          # Type: the Kernel PCA model

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
    MDS,
    classical_mds,      # perform classical MDS over a given distance matrix
    eigvals,            # eignenvalues of the transformation
    stress,             # stress evaluation

    gram2dmat, gram2dmat!,  # Gram matrix => Distance matrix
    dmat2gram, dmat2gram!,  # Distance matrix => Gram matrix

    ## lda
    LinearDiscriminant,     # Type: Linear Discriminant functional
    MulticlassLDAStats,     # Type: Statistics required for training multi-class LDA
    MulticlassLDA,          # Type: Multi-class LDA model
    SubspaceLDA,            # Type: LDA model for high-dimensional spaces

    ldacov,                 # Linear discriminant analysis based on covariances

    classweights,           # class-specific weights
    classmeans,             # class-specific means
    withclass_scatter,      # with-class scatter matrix
    betweenclass_scatter,   # between-class scatter matrix
    multiclass_lda_stats,   # compute statistics for multiclass LDA training
    multiclass_lda,         # train multi-class LDA based on statistics
    mclda_solve,            # solve multi-class LDA projection given sStatisticalModel

    ## ica
    ICA,                    # Type: the Fast ICA model

    icagfun,                # a function to get a ICA approx neg-entropy functor
    fastica!,               # core algorithm function for the Fast ICA

    ## fa
    FactorAnalysis,         # Type: the Factor Analysis model

    faem,                   # Maximum likelihood probabilistic PCA
    facm                    # EM algorithm for probabilistic PCA

    ## source files
    include("types.jl")
    include("common.jl")
    include("lreg.jl")
    include("whiten.jl")
    include("pca.jl")
    include("ppca.jl")
    include("kpca.jl")
    include("cca.jl")
    include("cmds.jl")
    include("lda.jl")
    include("ica.jl")
    include("fa.jl")

    ## deprecations
    @deprecate indim(f::Whitening) length(f::Whitening)
    @deprecate outdim(f::Whitening) length(f::Whitening)
    @deprecate indim(f::MulticlassLDA) size(f::MulticlassLDA)[1]
    @deprecate outdim(f::MulticlassLDA) size(f::MulticlassLDA)[2]
    @deprecate indim(f::SubspaceLDA) size(f::SubspaceLDA)[1]
    @deprecate outdim(f::SubspaceLDA) size(f::SubspaceLDA)[2]
    @deprecate indim(f::PCA) size(f::PCA)[1]
    @deprecate outdim(f::PCA) size(f::PCA)[2]
    @deprecate tvar(f::PCA) var(f::PCA) # total variance
    @deprecate transform(f::PCA, x) predict(f::PCA, x) #ex=false
    # @deprecate transform(m, x; kwargs...) predict(m, x; kwargs...) #ex=false
    # @deprecate transform(m; kwargs...) predict(m; kwargs...) #ex=false

end # module
