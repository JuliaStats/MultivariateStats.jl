module MultivariateAnalysis
    using NumericExtensions
    using MLBase

    import Base: show, dump

    # import & re-export symbols from MLBase
    import MLBase: indim, outdim, transform
    export indim, outdim, transform

    export 
        # pca
        PCA, pcacov, pcasvd, pca, reconstruct
        
    include("common.jl")
    include("pca.jl")
    
end # module
