tests = ["pca"]

println("Running tests:")

for t in tests
	fp = joinpath("test", "$(t).jl")
    println(" * $(fp)")
    include(fp)
end
