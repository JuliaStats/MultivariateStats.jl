abstract type AbstractDimensionalityReduction <: RegressionModel end

"""
    size(model::AbstractDimensionalityReduction, d::Int)

Returns the dimension of the input data if `d == 1`, the dimension of the output data
if `d == 2`, otherwise throws error.
"""
function size(model::AbstractDimensionalityReduction, d::Integer)
    dims = size(model)
    @assert length(dims) >= d "Cannot access dimensional information"
    return dims[d]
end

"""
    projection(model::AbstractDimensionalityReduction)

Return the projection matrix of the model.
"""
projection(model::AbstractDimensionalityReduction) = error("'projection' is not defined for $(typeof(model)).")

"""
    reconstruct(model::AbstractDimensionalityReduction, y)

Return the model response (a.k.a. the dependent variable).
Throws an `ArgumentError` if `y` is not an `AbstractVecOrMat` of real numbers, indicating the invalid 
input type and the expected type.
"""
function reconstruct(model::AbstractDimensionalityReduction, y)
    throw(ArgumentError("Invalid input type $(typeof(y)) for reconstruct(::$(typeof(model))). " * 
    "Expected an AbstractVecOrMat of real numbers."
    ))
end

abstract type LinearDimensionalityReduction <: AbstractDimensionalityReduction end

"""
    loadings(model::LinearDimensionalityReduction)

Return the model loadings (a.k.a. eigenvectors scaled up by the variances).
"""
loadings(model::LinearDimensionalityReduction) = error("'loadings' is not defined for $(typeof(model)).")

abstract type NonlinearDimensionalityReduction <: AbstractDimensionalityReduction end
abstract type LatentVariableDimensionalityReduction <: AbstractDimensionalityReduction end
