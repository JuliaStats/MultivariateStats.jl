abstract type AbstractDimensionalityReduction <: RegressionModel end

"""
    projection(model::AbstractDimensionalityReduction)

Return the projection matrix of the model.
"""
projection(model::AbstractDimensionalityReduction) = error("'projection' is not defined for $(typeof(model)).")

"""
    reconstruct(model::AbstractDimensionalityReduction, y)

Return the model response (a.k.a. the dependent variable).
"""
reconstruct(model::AbstractDimensionalityReduction, y) = error("'reconstruct' is not defined for $(typeof(model)).")

abstract type LinearDimensionalityReduction <: AbstractDimensionalityReduction end

"""
    loadings(model::LinearDimensionalityReduction)

Return the model loadings (a.k.a. eigenvectors scaled up by the variances).
"""
loadings(model::LinearDimensionalityReduction) = error("'loadings' is not defined for $(typeof(model)).")

abstract type NonlinearDimensionalityReduction <: AbstractDimensionalityReduction end
abstract type LatentVariableDimensionalityReduction <: AbstractDimensionalityReduction end
