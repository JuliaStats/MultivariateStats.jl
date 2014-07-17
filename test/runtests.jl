tests = ["pca", 
         "cca", 
         "cmds", 
         "lda"]

println("Running tests:")

for t in tests
    fp = string(t, ".jl")
    println(" * $(fp)")
    include(fp)
end
