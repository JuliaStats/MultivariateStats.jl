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
         "fa"
         ]

for test in tests
    include(test*".jl")
end
