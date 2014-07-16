
# print arrays in pretty way

function printarr(io::IO, a::AbstractArray)
    Base.with_output_limit(()->Base.showarray(io, a, header=false, repr=false))
end

printvec(io::IO, a::AbstractVector) = printarr(io, a')

printarrln(io::IO, a::AbstractArray) = (printarr(io, a); println(io))
printvecln(io::IO, a::AbstractVector) = (printvec(io, a); println(io))

# centralize 

centralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x - m)::typeof(x)
centralize(x::AbstractMatrix, m::AbstractVector) = (isempty(m) ? x : x .- m)::typeof(x)

decentralize(x::AbstractVector, m::AbstractVector) = (isempty(m) ? x : x + m)::typeof(x)
decentralize(x::AbstractMatrix, m::AbstractVector) = (isempty(m) ? x : x .+ m)::typeof(x)

# get a full mean vector 

fullmean{T}(d::Int, mv::Vector{T}) = (isempty(mv) ? zeros(T, d) : mv)::Vector{T} 

# symmmetrize a matrix

function symmetrize!(A::Matrix)
    n = size(A, 1)
    @assert size(A, 2) == n
    for j = 1:n
        for i = 1:j-1
            @inbounds A[i,j] = A[j,i]
        end
        for i = j+1:n
            @inbounds A[i,j] = middle(A[i,j], A[j,i])
        end
    end
    return A
end

# percolumn dot

function coldot(X::Matrix, Y::Matrix)
    m = size(X, 1)
    n = size(X, 2)
    @assert size(Y) == (m, n)
    R = zeros(n)
    for j = 1:n
        R[j] = dot(view(X,:,j), view(Y,:,j))
    end
    return R
end

