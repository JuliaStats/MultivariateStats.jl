# Canonical Correlation Analysis

#### CCA Type

type CCA
    xmean::Vector{Float64}  # sample mean of X: of length dx (can be empty)
    ymean::Vector{Float64}  # sample mean of Y: of length dy (can be empty)  
    xproj::Matrix{Float64}  # projection matrix for X, of size (dx, p)
    yproj::Matrix{Float64}  # projection matrix for Y, of size (dy, p)
    corrs::Vector{Float64}  # correlations, of length p

    function CCA(xm::Vector{Float64}, 
                 ym::Vector{Float64}, 
                 xp::Matrix{Float64},
                 yp::Matrix{Float64}, 
                 crs::Vector{Float64})

        dx, px = size(xp)
        dy, py = size(yp)

        isempty(xm) || length(xm) == dx ||
            throw(DimensionMismatch("Incorrect length of xmean."))

        isempty(ym) || length(ym) == dy ||
            throw(DimensionMismatch("Incorrect length of ymean."))

        px == py || 
            throw(DimensionMismatch("xproj and yproj should have the same number of columns."))

        length(crs) == px ||
            throw(DimensionMismatch("Incorrect length of corrs."))

        new(xm, ym, xp, yp, crs)
    end
end

## properties

xindim(M::CCA) = size(M.xproj, 1)
yindim(M::CCA) = size(M.yproj, 1)
outdim(M::CCA) = size(M.xproj, 2)

xmean(M::CCA) = fullmean(xindim(M), M.xmean)
ymean(M::CCA) = fullmean(yindim(M), M.ymean)

xprojection(M::CCA) = M.xproj
yprojection(M::CCA) = M.yproj

correlations(M::CCA) = M.corrs

## use

xtransform{T<:Real}(M::CCA, X::AbstractVecOrMat{T}) = At_mul_B(M.xproj, centralize(X, M.xmean))
ytransform{T<:Real}(M::CCA, Y::AbstractVecOrMat{T}) = At_mul_B(M.yproj, centralize(Y, M.ymean))



