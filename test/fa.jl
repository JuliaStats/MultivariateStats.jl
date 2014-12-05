# Test factor analysis implementation using a test similar to the one in
# scikit-learn, so we can directly compare implementations if needed.

using MultivariateStats

n_samples = 20
n_features = 5
n_components = 3

X = readdlm("X_fa.csv", ',')'
fa, L = fit(FactorAnalysis, X, n_components)

X_t = transform(fa, X)

X_rec = fa.Vbar .+ fa.F*X_t .+ fa.Ψ.*randn(n_features,n_samples)
