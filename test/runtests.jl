tests = ["pca", 
         "cca", 
         "cmds"]

println("Running tests:")

for t in tests
    fp = string(t, ".jl")
    println(" * $(fp)")
    include(fp)
end
