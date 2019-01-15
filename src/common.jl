
# print arrays in pretty way

function printarr(io::IO, a::AbstractArray)
    Base.with_output_limit(()->Base.showarray(io, a, header=false, repr=false))
end

printvec(io::IO, a::AbstractVector) = printarr(io, a')

printarrln(io::IO, a::AbstractArray) = (printarr(io, a); println(io))
printvecln(io::IO, a::AbstractVector) = (printvec(io, a); println(io))

# centralize

centralize(x, m::AbstractVector) = (isempty(m) ? x : x .- m)
decentralize(x, m::AbstractVector) = (isempty(m) ? x : x .+ m)
centralize!(x, m::AbstractVector) = (isempty(m) || (x .= x .- m); x)
decentralize!(x, m::AbstractVector) = (isempty(m) || (x .= x .+ m); x)

# standardize

standardize(x, s::AbstractVector) = (isempty(s) ? x : x ./ s)
destandardize(x, s::AbstractVector) = (isempty(s) ? x : x .* s)
standardize!(x, s::AbstractVector) = (isempty(s) || (x .= x ./ s); x)
destandardize!(x, s::AbstractVector) = (isempty(s) || (x .= x .* s); x)

# z transform
function ztransform!(x, m::AbstractVector, s::AbstractVector)
    if isempty(m) || isempty(s)
        centralize!(x, m)
        standardize!(x, s)
    else
        x .= (x .- m) ./ s
    end
    return x
end
ztransform(x, m, s) = ztransform!(copy(x), m, s)

function deztransform!(x, m::AbstractVector, s::AbstractVector)
    if isempty(m) || isempty(s)
        destandardize!(x, s)
        decentralize!(x, m)
    else
        x .= (x .* s) .+ m
    end
end
deztransform(x, m, s) = deztransform!(copy(x), m, s)

# get a full mean/std vector

fullmean(d::Int, mv::Vector{T}) where T = (isempty(mv) ? zeros(T, d) : mv)::Vector{T}
fullstd(d::Int, sv::Vector{T}) where T = (isempty(sv) ? ones(T, d) : sv)::Vector{T}

preprocess_mean(X::AbstractMatrix{T}, m) where T<:Real =
    (m == nothing ? vec(mean(X, dims=2)) : m == 0 ? T[] :  m)::Vector{T}

preprocess_std(X::AbstractMatrix{T}, s, m = nothing) where T<:Real =
    (s == nothing ? vec(std(X, mean = m, dims = 2)) : s == 1 ? T[] : s)::Vector{T}

# choose the first k values and columns
#
# S must have fields: values & vectors

function extract_kv(fac::Factorization{T}, ord::AbstractVector{Int}, k::Int) where T
    si = ord[1:k]
    vals = fac.values[si]::Vector{T}
    vecs = fac.vectors[:, si]::Matrix{T}
    return (vals, vecs)
end


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

function coldot(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T<:Real
    m = size(X, 1)
    n = size(X, 2)
    @assert size(Y) == (m, n)
    R = zeros(T, n)
    for j = 1:n
        R[j] = dot(view(X,:,j), view(Y,:,j))
    end
    return R
end

# qnormalize!

function qnormalize!(X, C)
    # normalize each column of X (say x), such that x'Cx = 1
    m = size(X, 1)
    n = size(X, 2)
    CX = C * X
    for j = 1:n
        x = view(X,:,j)
        cx = view(CX,:,j)
        rmul!(x, inv(sqrt(dot(x, cx))))
    end
    return X
end

# add_diag!

function add_diag!(A::AbstractMatrix, v::Real)
    # add v to diagonal of A
    m = size(A, 1)
    n = size(A, 2)
    @assert m == n
    if v != zero(v)
        for i = 1:n
            @inbounds A[i,i] += v
        end
    end
    return A
end

# regularize a symmetric matrix
function regularize_symmat!(A::Matrix{T}, lambda::Real) where T<:Real
    if lambda > 0
        emax = eigmax(Symmetric(A))
        add_diag!(A, emax * lambda)
    end
    return A
end

