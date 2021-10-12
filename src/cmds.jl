# Classical Multidimensional Scaling

## convert Gram matrix to Distance matrix

function gram2dmat!(D::AbstractMatrix{DT}, G::AbstractMatrix) where {DT<:Real}
    # argument checking
    m = size(G, 1)
    n = size(G, 2)
    m == n || error("D should be a square matrix.")
    size(D) == (m, n) ||
        throw(DimensionMismatch("Sizes of D and G do not match."))

    # implementation
    for j = 1:n
        for i = 1:j-1
            @inbounds D[i,j] = D[j,i]
        end
        D[j,j] = zero(DT)
        for i = j+1:n
            @inbounds D[i,j] = sqrt(G[i,i] + G[j,j] - 2 * G[i,j])
        end
    end
    return D
end

gram2dmat(G::AbstractMatrix{T}) where {T<:Real} = gram2dmat!(similar(G, T), G)

## convert Distance matrix to Gram matrix

function dmat2gram!(G::AbstractMatrix{GT}, D::AbstractMatrix) where GT
    # argument checking
    n = LinearAlgebra.checksquare(D)
    size(G) == (n, n) ||
        throw(DimensionMismatch("Sizes of G and D do not match."))

    # implementation
    u = zeros(GT, n)
    s = 0.0
    for j = 1:n
        s += (u[j] = sum(abs2, view(D,:,j)) / n)
    end
    s /= n

    for j = 1:n
        for i = 1:j-1
            @inbounds G[i,j] = G[j,i]
        end
        for i = j:n
            @inbounds G[i,j] = (u[i] + u[j] - abs2(D[i,j]) - s) / 2
        end
    end
    return G
end

momenttype(T) = typeof((zero(T) * zero(T) + zero(T) * zero(T))/ 2)
dmat2gram(D::AbstractMatrix{T}) where {T<:Real} = dmat2gram!(similar(D, momenttype(T)), D)

## Classical MDS

"""Classical MDS type"""
struct MDS{T<:Real}
    d::Real                  # original dimension
    X::AbstractMatrix{T}     # fitted data, X (d x n)
    λ::AbstractVector{T}     # eigenvalues in feature space, (k x 1)
    U::AbstractMatrix{T}     # eigenvectors in feature space, U (n x k)
end

## properties

indim(M::MDS) = M.d
outdim(M::MDS) = size(M.U,2)

projection(M::MDS) = M.U
eigvals(M::MDS) = M.λ

loadings(M::MDS) = eigvals(M)' .* projection(M)

## use

"""Calculate out-of-sample multidimensional scaling transformation"""
function transform(M::MDS{T}, x::AbstractVector{<:Real}; distances=false) where {T<:Real}
    d = if isnan(M.d) # model has only distance matrix
        @assert distances "Cannot transform points if model was fitted with a distance matrix. Use point distances."
        size(x, 1) != size(M.X, 1) && throw(
            DimensionMismatch("Point distances should be calculated to all original points"))
        x
    else
        if distances
            size(x, 1) != size(M.X, 2) && throw(
                DimensionMismatch("Point distances should be calculated to all original points."))
            x
        else
            size(x, 1) != size(M.X, 1) && throw(
                DimensionMismatch("Points and original data must have same dimensionality."))
            pairwise((x,y)->norm(x-y), M.X, x)
        end
    end

    # get distance matrix
    D = isnan(M.d) ? M.X : pairwise((x,y)->norm(x-y), eachcol(M.X), symmetric=true)
    d = d.^2

    # b = 0.5*(ones(n,n)*d./n - d + D*ones(n,1)./n - ones(n,n)*D*ones(n,1)./n^2)
    mD = mean(D.^2, dims=2)
    b = (d  .- mean(d, dims=1) .- mD .+ mean(mD)) ./ -2

    # sqrt(λ)⁻¹U'b
    λ = vcat(M.λ, zeros(T, outdim(M) - length(M.λ)))
    return M.U' * b ./ sqrt.(λ)
end

"""Calculate multidimensional scaling transformation"""
function transform(M::MDS{T}) where {T<:Real}
    # if there are non-positive missing eigval then padd with zeros
    λ = vcat(M.λ, zeros(T, outdim(M) - length(M.λ)))
    return diagm(0=>sqrt.(λ)) * M.U'
end

## show

function Base.show(io::IO, M::MDS)
    print(io, "Classical MDS(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## interface functions

"""

Compute an embedding of points by classical multidimensional scaling (MDS).
There are two calling options, specified via the required keyword argument `distances`:

    mds = fit(MDS, X; distances=false, maxdim=size(X,1)-1)

`X` is the data matrix. Distances between pairs of columns of `X` are computed using the Euclidean norm.
This is equivalent to performing PCA on `X`.

    mds = fit(MDS, D; distances=true,  maxdim=size(D,1)-1)

`D` is a symmetric matrix `D` of distances between points.
"""
function fit(::Type{MDS}, X::AbstractMatrix{T};
             maxoutdim::Int = size(X,1)-1,
             distances::Bool) where T<:Real

    # get distance matrix and space dimension
    D, d = if !distances
        pairwise((x,y)->norm(x-y), eachcol(X), symmetric=true), size(X,1)
    else
        X, NaN
    end
    G = dmat2gram(D)

    n = size(D, 1)
    m = min(maxoutdim, n) #Actual number of eigenpairs wanted

    E = eigen!(Hermitian(G))

    #Sometimes dmat2gram produces a negative definite matrix, and the sign just
    #needs to be flipped. The heuristic to check for this robustly is to check
    #if there is a negative eigenvalue of magnitude larger than the largest
    #positive eigenvalue, and flip the sign of eigenvalues if necessary.
    mineig, maxeig = extrema(E.values)
    if mineig < 0 && abs(mineig) > abs(maxeig)
        #do flip
        ord = sortperm(E.values)
        λ = -E.values[ord[1:m]]
    else
        ord = sortperm(E.values; rev=true)
        λ = E.values[ord[1:m]]
    end

    for i in 1:m
        if λ[i] <= 0
            #Keeping all remaining eigenpairs would not change solution (if 0)
            #or make the answer _worse_ (if <0).
            #The least squares solution would want to throw all these away.
            @warn("Gramian has only $(i-1) positive eigenvalue(s)")
            m = i-1
            ord = ord[1:m]
            λ = λ[1:m]
            break
        end
    end

    #Check if the last considered eigenvalue is degenerate
    if m>0
        nevalsmore = sum(abs.(E.values[ord[m+1:end]] .- λ[m]) .< n*eps())
        nevals = sum(abs.(E.values .- λ[m]) .< n*eps())
        if nevalsmore > 1
            @warn("The last eigenpair is degenerate with $(nevals-1) others; $nevalsmore were ignored. Answer is not unique")
        end
    end
    U = E.vectors[:, ord[1:m]]

    #Add trailing zero coordinates if dimension of embedding space (p) exceeds
    #number of eigenpairs used (m)
    if m < maxoutdim
        U = [U zeros(T, n, maxoutdim-m)]
    end

    return MDS(d, X, λ, U)
end

@deprecate classical_mds(D::AbstractMatrix, p::Int) transform(fit(MDS, D, maxoutdim=p, distances=true))

function stress(M::MDS)
    # calculate distances if original data was stored
    DX = isnan(M.d) ? M.X : pairwise((x,y)->norm(x-y), eachcol(M.X), symmetric=true)
    DY = pairwise((x,y)->norm(x-y), eachcol(transform(M)), symmetric=true)
    n = size(DX,1)
    return sqrt(2*sum((DX - DY).^2)/sum(DX.^2));
end
