## Geometric Utilities ##
export min_dist, interior_dist, signed_dist, extended_bbox

# Implement CBPQ support functions for Meshes.jl types
CBPQ.support(g::Geometry, dir::AbstractVector) =
    SVector(Meshes.supportfun(g, Vec(dir)) |> coordinates)
CBPQ.support(p::Point, dir::AbstractVector) =
    SVector(coordinates(p))
CBPQ.support(p::AbstractVector, dir::AbstractVector) =
    SVector{length(p)}(p)

# Check whether a point represented as a vector is in a particular geometry or mesh
Base.in(p::AbstractVector{<:Real}, g::Geometry) =
    Point(p) in g
Base.in(p::AbstractVector{<:Real}, d::Domain) =
    Point(p) in d
Base.in(p::AbstractVector{<:Real}, b::Box{D}) where {D} =
    length(p) == D && all(b.min.coords[i] <= p[i] <= b.max.coords[i] for i in 1:D)

# Convenience union types
const GeometryOrMesh{D, T} = Union{Geometry{D, T}, Mesh{D, T}}
const PointOrGeometryOrMesh{D, T} = Union{PointOrGeometry{D, T}, Mesh{D, T}}

"Difference between centroids of two points or geometries or meshes."
centroid_diff(g1::PointOrGeometryOrMesh, g2::PointOrGeometryOrMesh) =
    centroid(g1) - centroid(g2)
centroid_diff(p::AbstractVector, g::PointOrGeometryOrMesh) =
    p - coordinates(centroid(g))

# Define reverse AD rule for `centroid_diff`
function ChainRulesCore.rrule(
    ::typeof(centroid_diff), p::AbstractVector{T}, g::PointOrGeometryOrMesh
) where {T}
    diff = centroid_diff(p, g)
    function centroid_diff_pullback(diff_)
        p_ = Matrix{T}(I, length(p), length(p))' * diff_
        return NoTangent(), p_, NoTangent()
    end
    return diff, centroid_diff_pullback
end

"Returns minimum translation vector between two convex bodies `p` and `q`."
function min_translation(
    p::Any, q::Any, init_dir::SVector{D, T};
    max_iter=100, atol::T=sqrt(eps(T))*oneunit(T)) where {D, T}
    collision, dir, psimplex, qsimplex, sz =
        CBPQ.gjk(p, q, init_dir, max_iter, atol, CBPQ.minimum_distance_cond)
    return collision ? zero(dir) : dir
end

min_translation(p::PointOrGeometry, q::PointOrGeometry; kwargs...) =
    min_translation(p, q, SVector(centroid_diff(p, q)); kwargs...)
min_translation(p::AbstractVector, q::PointOrGeometry; kwargs...) =
    min_translation(p, q, SVector(centroid_diff(p, q)); kwargs...)

"Returns minimum translation between a convex body `p` and mesh `q`."
function min_translation(p::AbstractVector, q::Mesh; kwargs...)
    min_dist = Inf
    min_dir = nothing
    for element in q
        dir = min_translation(p, element; kwargs...)
        dist = norm(diff)
        if dist < min_dist
            min_dist = dist
            min_dir = dir
        end
    end
    return min_dir
end

"Return minimum distance to a point, geometry, or mesh."
min_dist(g1::PointOrGeometry, g2::PointOrGeometry) = 
    norm(min_translation(g1, g2))
min_dist(p::AbstractVector, g::PointOrGeometryOrMesh) =
    norm(min_translation(p, g))

# Define reverse AD rule for `min_dist`
function ChainRulesCore.rrule(
    ::typeof(min_dist), p::AbstractVector, g::PointOrGeometryOrMesh
)
    dir = min_translation(p, g)
    dist = norm(dir)
    function min_dist_pullback(dist_)
        p_ = dist_ * (-dir ./ dist)
        return NoTangent(), p_, NoTangent()
    end
    return dist, min_dist_pullback
end

"Compute distance from an interior point to the boundary of a geometry."
interior_dist(p, g::Geometry) = 
    minimum(min_dist(p, s) for s in simplexify(boundary(g)))
interior_dist(p, b::Union{Ball,Sphere}) =
    b.radius - norm(centroid_diff(p, b))
interior_dist(p, m::Mesh) = 
    minimum(min_dist(p, el) for el in m)

# Define reverse AD rules for `interior_dist`
function ChainRulesCore.rrule(
    ::typeof(interior_dist), p::AbstractVector, g::GeometryOrMesh
)
    min_dist = Inf
    min_dir = nothing
    elements = g isa Geometry ? simplexify(boundary(g)) : g
    for el in elements
        dir = min_translation(p, el)
        dist = norm(dir)
        if dist < min_dist
            min_dist = dist
            min_dir = dir
        end
    end
    function interior_dist_pullback(dist_)
        p_ = dist_ * (-min_dir ./ min_dist)
        return NoTangent(), p_, NoTangent()
    end
    return min_dist, interior_dist_pullback
end

function ChainRulesCore.rrule(
    ::typeof(interior_dist), p::AbstractVector, b::Union{Ball,Sphere}
)
    diff = centroid_diff(p, b)
    diff_norm = norm(diff)
    dist = b.radius - diff_norm
    function interior_dist_pullback(dist_)
        p_ = dist_ * (-diff ./ diff_norm)
        return NoTangent(), p_, NoTangent()
    end
    return dist, interior_dist_pullback
end

"Return signed distance of a point to a geometry or mesh."
function signed_dist(p::AbstractVector, g::GeometryOrMesh)
    g = ignore_derivatives(g)
    inside = @ignore_derivatives p in g
    return inside ? -interior_dist(p, g) : min_dist(p, g)
end
signed_dist(p::Point, g::GeometryOrMesh) =
    p in g ? -interior_dist(p, g) : min_dist(p, g)

function signed_dist(p, gs::AbstractVector{<:GeometryOrMesh})
    if isempty(gs)
        return Inf
    else
        dists = map(g -> signed_dist(p, g)::Float64, ignore_derivatives(gs))
        return minimum(dists)
    end
end
    
function extended_bbox(g::GeometryOrMesh{D}, d_safe::Real) where {D}
    box = boundingbox(g)
    offset = Vec(ntuple(Returns(d_safe), Val(D)))
    return Box(box.min - offset, box.max + offset)
end
