tests = ["pca", 
         "cca", 
         "cmds", 
         "lda", 
         "mclda"]

println("Running tests:")

for t in tests
    fp = string(t, ".jl")
    println(" * $(fp)")
    include(fp)
end
