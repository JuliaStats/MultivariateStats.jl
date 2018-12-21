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
    D::AbstractMatrix{T}     # fitted data, X (d x d)
    λ::AbstractVector{T}     # sqrt. eigenvalues in feature space, √λ (k x 1)
    U::AbstractMatrix{T}     # eigenvectors in feature space, U (n x k)
end

## properties

indim(M::MDS) = M.d
outdim(M::MDS) = size(M.U,2)

projection(M::MDS) = M.U
eigvals(M::MDS) = M.λ

## use

function transform(M::MDS)
    return Diagonal(sqrt.(M.λ)) * M.U'
end

## show

function Base.show(io::IO, M::MDS)
    print(io, "Classical MDS(indim = $(indim(M)), outdim = $(outdim(M)))")
end

## interface functions

function fit(::Type{MDS}, X::AbstractMatrix{T};
             maxoutdim::Int = size(X,1)-1,
             distances=false) where T<:Real

    # get distance matrix and space dimension
    D, d = if !distances
        pairwise((x,y)->norm(x-y), X), size(X,1)
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
            dowarn && @warn("The last eigenpair is degenerate with $(nevals-1) others; $nevalsmore were ignored. Answer is not unique")
        end
    end
    U = E.vectors[:, ord[1:m]]

    #Add trailing zero coordinates if dimension of embedding space (p) exceeds
    #number of eigenpairs used (m)
    if m < maxoutdim
        U = [U zeros(T, n, maxoutdim-m)]
    end

    return MDS(d, D, λ, U)
end

@deprecate classical_mds(D::AbstractMatrix, p::Int) transform(fit(MDS, D, maxoutdim=p, distances=true))
