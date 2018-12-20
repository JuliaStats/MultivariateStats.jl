using MultivariateStats
using Test

tests = ["lreg",
         "whiten",
         "pca",
         "cca",
         "cmds",
         "lda",
         "mclda",
         "ica",
         "ppca",
         "kpca",
         "fa"]

@testset for test in tests
    include(test*".jl")
end
