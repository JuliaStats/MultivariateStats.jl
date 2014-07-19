tests = ["whiten",
         "pca", 
         "cca", 
         "cmds", 
         "lda", 
         "mclda", 
         "fastica"]

println("Running tests:")

for t in tests
    fp = string(t, ".jl")
    println(" * $(fp)")
    include(fp)
end
