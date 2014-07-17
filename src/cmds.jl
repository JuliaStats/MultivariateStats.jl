# Classical Multidimensional Scaling

## convert Gram matrix to Distance matrix

function gram2dmat!{DT}(D::AbstractMatrix{DT}, G::AbstractMatrix)
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

gram2dmat{T<:Real}(G::AbstractMatrix{T}) = gram2dmat!(similar(G, Base.momenttype(T)), G)

## convert Distance matrix to Gram matrix

function dmat2gram!{GT}(G::AbstractMatrix{GT}, D::AbstractMatrix)
    # argument checking
    m = size(D, 1)
    n = size(D, 2)
    m == n || error("D should be a square matrix.")
    size(G) == (m, n) ||
        throw(DimensionMismatch("Sizes of G and D do not match."))

    # implementation
    u = zeros(GT, n)
    s = 0.0
    for j = 1:n
        s += (u[j] = Base.sumabs2(view(D,:,j)) / n)
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

dmat2gram{T<:Real}(D::AbstractMatrix{T}) = dmat2gram!(similar(D, Base.momenttype(T)), D)

## classical MDS

function classical_mds{T<:Real}(D::AbstractMatrix{T}, p::Integer)
    n = size(D, 1)
    p < n || error("p must be less than n.")

    G = dmat2gram(D)
    E = eigfact!(Symmetric(G))
    ord = sortperm(E.values; rev=true)
    (v, U) = extract_kv(E, ord, p)
    for i = 1:p
        @inbounds v[i] = sqrt(v[i])
    end
    scale!(U, v)'
end


