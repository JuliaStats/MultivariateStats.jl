tests = ["pca",
         "common",
         "lreg",
         "whiten",
         "cca",
         "cmds",
         "lda",
         "mclda",
         "ica",
         "ppca",
         "kpca",
         "fa"]

for test in tests
    include(test*".jl")
end
