# MultivariateStats.jl Documentation

```@meta
CurrentModule = MultivariateStats
DocTestSetup = quote
    using Statistics
    using Random
end
```

[MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) is a Julia package for multivariate statistical analysis. It provides a rich set of useful analysis techniques, such as PCA, CCA, LDA, ICA, etc.

```@contents
Pages = ["whiten.md", "lreg.md", "lda.md", "pca.md", "ica.md", "cca.md", "fa.md", "mds.md", "api.md"]
Depth = 2
```

**Notes:** All methods implemented in this package adopt the column-major convention of JuliaStats: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).
