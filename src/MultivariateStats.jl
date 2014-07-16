module MultivariateStats
    using StatsBase

    import Base: length, show

    export 

    # common
    indim, outdim, transform, reconstruct,

    # pca
    PCA, pca, pcacov, pcastd

    
    ## source files
    include("pca.jl")

end # module
