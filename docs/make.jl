using Documenter, MultivariateStats, StatsBase, Statistics, Random, LinearAlgebra

makedocs(
    sitename = "MultivariateStats.jl",
    modules = [MultivariateStats],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliastats.org/MultivariateStats.jl/stable/"
    ),
    pages = ["Home"=>"index.md",
             "whiten.md",
             "lreg.md",
             "lda.md",
             "pca.md",
             "ica.md",
             "cca.md",
             "fa.md",
             "mds.md",
             "Development"=>"api.md"]
)

deploydocs(
    repo = "github.com/JuliaStats/MultivariateStats.jl.git"
)
