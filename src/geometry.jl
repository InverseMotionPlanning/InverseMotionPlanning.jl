## Geometric Utilities ##
export min_dist, interior_dist, signed_dist, extended_bbox, Scene

# Implement CBPQ support functions for Meshes.jl types
CBPQ.support(g::Geometry, dir::AbstractVector) =
    SVector(Meshes.supportfun(g, Vec(dir)) |> coordinates)
CBPQ.support(p::Point, dir::AbstractVector) =
    SVector(coordinates(p))
CBPQ.support(p::AbstractVector, dir::AbstractVector) =
    SVector{length(p)}(p)

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

"Difference between centroids of two points or geometries."
centroid_diff(g1::PointOrGeometry, g2::PointOrGeometry) =
    centroid(g1) - centroid(g2)
centroid_diff(p::AbstractVector, g::PointOrGeometry) =
    p - coordinates(centroid(g))

# Define reverse AD rule for `centroid_diff`
function ChainRulesCore.rrule(
    ::typeof(centroid_diff), p::AbstractVector{T}, g::PointOrGeometry
) where {T}
    diff = centroid_diff(p, g)
    function centroid_diff_pullback(diff_)
        p_ = Matrix{T}(I, length(p), length(p))' * diff_
        return NoTangent(), p_, NoTangent()
    end
    return diff, centroid_diff_pullback
end

"Return minimum distance between two points or geometries."
min_dist(g1::PointOrGeometry, g2::PointOrGeometry) = 
    norm(min_translation(g1, g2))
min_dist(p::AbstractVector, g::PointOrGeometry) =
    norm(min_translation(p, g))

# Define reverse AD rule for `min_dist`
function ChainRulesCore.rrule(
    ::typeof(min_dist), p::AbstractVector, g::PointOrGeometry
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

# Define reverse AD rules for `interior_dist`
function ChainRulesCore.rrule(
    ::typeof(interior_dist), p::AbstractVector, g::Geometry
)
    min_dist = Inf
    min_dir = nothing
    for s in simplexify(boundary(g))
        dir = min_translation(p, s)
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

function signed_dist(p::AbstractVector, g::Geometry)
    g = ignore_derivatives(g)
    inside = @ignore_derivatives Point(p) in g
    return inside ? -interior_dist(p, g) : min_dist(p, g)
end
signed_dist(p::Point, g::Geometry) =
    p in g ? -interior_dist(p, g) : min_dist(p, g)

function signed_dist(p, gs::AbstractVector{<:Geometry})
    if isempty(gs)
        return Inf
    else
        dists = map(g -> signed_dist(p, g)::Float64, ignore_derivatives(gs))
        return minimum(dists)
    end
end
    
function extended_bbox(g::Geometry{D}, d_safe::Real) where {D}
    box = boundingbox(g)
    offset = Vec(ntuple(Returns(d_safe), Val(D)))
    return Box(box.min - offset, box.max + offset)
end

## Scene Datatype ##

struct Scene{D, T} <: Meshes.Domain{D, T}
    obstacles::Vector{Geometry{D, T}}   
end

Scene(obstacles::Geometry{D,T}...) where {D, T} =
    Scene{D,T}(collect(obstacles))

Meshes.element(scene::Scene, ind) = scene.obstacles[ind]
Meshes.nelements(scene::Scene) = length(scene.obstacles)
Base.eltype(scene::Scene) = eltype(scene.obstacles)

function signed_dist(p, scene::Scene)
    dists = map(obs -> signed_dist(p, obs), ignore_derivatives(scene.obstacles))
    return minimum(dists)
end

function possible_collisions(p, scene::Scene{D, T}, d_safe::Real) where {D, T}
    obstacles = filter(scene.obstacles) do obs
        return @ignore_derivatives Point(p) in extended_bbox(obs, d_safe)        
    end::Vector{Geometry{D, T}}
    return obstacles
end