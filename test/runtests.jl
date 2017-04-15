tests = ["lreg",
         "whiten",
         "pca",
         "cca",
         "cmds",
         "lda",
         "mclda",
         "ica",
         "ppca",
         "fa"]

println("Running tests:")

for t in tests
    fp = string(t, ".jl")
    println(" * $(fp)")
    include(fp)
end
