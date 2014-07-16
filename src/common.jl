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
