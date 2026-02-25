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
         "fa",
         "mca",
         ]

for test in tests
    include(test*".jl")
end
