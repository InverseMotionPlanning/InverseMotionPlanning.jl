## Scene Datatype ##

export Scene, sceneviz

"""
    Scene

Represents a scene of immovable obstacles, movable objects, and distinguished
regions, with optional scene limits.
"""
@kwdef struct Scene{D, T}
    "Immovable obstacles."
    obstacles::Vector{GeometryOrMesh{D, T}} = []
    "Named movable objects."
    objects::Dict{Symbol, GeometryOrMesh{D, T}} = Dict()
    "Named regions."
    regions::Dict{Symbol, Geometry{D, T}} = Dict()
    "Scene limits."
    limits::Box{D, T} = Box(Point(fill(-Inf, D)), Point(fill(Inf, D)))
end

Scene{D}(args...; kwargs...) where {D} = Scene{D, Float64}(args...; kwargs...)

Scene(obstacles::GeometryOrMesh{D,T}...; kwargs...) where {D, T} =
    Scene{D,T}(;obstacles=collect(GeometryOrMesh{D,T}, obstacles), kwargs...)

# Geometry functions

Meshes.paramdim(::Scene{D}) where {D} = D

"Return signed distance of a point to any collider in a scene."
function signed_dist(p, scene::Scene)
    all_colliders = [scene.obstacles; collect(values(scene.objects))]
    return signed_dist(p, all_objects)
end

"""
Return all obstacles or objects that might collide with a point given a
minimum safe distance `d_safe`.
"""
function possible_colliders(p, scene::Scene{D, T}, d_safe::Real) where {D, T}
    obstacles = filter(scene.obstacles) do obs
        return @ignore_derivatives p in extended_bbox(obs, d_safe)        
    end::Vector{GeometryOrMesh{D, T}}
    isempty(scene.objects) && return obstacles
    objects = Iterators.filter(values(scene.objects)) do obj
        return @ignore_derivatives p in extended_bbox(obj, d_safe)
    end
    objects = collect(GeometryOrMesh{D, T}, objects)
    return append!(obstacles, objects)
end

# Pretty printing

function Base.show(io::IO, ::MIME"text/plain", scene::Scene)
    print(io, summary(scene))
    # Print obstacles
    N = length(scene.obstacles)
    if N > 0
        print(io, "\n", "  obstacles: ")
        I, J = N > 10 ? (5, N-4) : (N, N+1)
        for i in 1:N
            I < i < J && continue
            print(io, "\n", "    └─$(scene.obstacles[i])")
        end
    end
    # Print objects
    N = length(scene.objects)
    if N > 0
        print(io, "\n", "  objects: ")
        I, J = N > 10 ? (5, N-4) : (N, N+1)
        for (i, (name, obj)) in enumerate(scene.objects)
            I < i < J && continue
            print(io, "\n", "    └─$name: $obj")
        end
    end
    # Print regions
    N = length(scene.regions)
    if N > 0
        print(io, "\n", "  regions: ")
        I, J = N > 10 ? (5, N-4) : (N, N+1)
        for (i, (name, region)) in enumerate(scene.regions)
            I < i < J && continue
            print(io, "\n", "    └─$name: $region")
        end
    end
    # Print scene limits
    print(io, "\n", "  limits: ")
    print(io, "\n", "    min: $(coordinates(scene.limits.min))")
    print(io, "\n", "    max: $(coordinates(scene.limits.max))")
end

# Plotting recipe

@Makie.recipe(SceneViz, scene) do makie_scene
    Makie.Attributes(;
      size           = Makie.theme(makie_scene, :markersize),
      obstacle_color = :grey,
      obstacle_colorscheme = nothing,      
      object_color = MeshViz.colorschemes[:Set1_9][1:9],
      object_colorscheme = :Set1_9,
      region_color = MeshViz.colorschemes[:Pastel1_9][1:9],
      region_colorscheme = :Pastel1_9,
      colorscheme    = :viridis,
      alpha          = 1.0,
      facetcolor     = :gray30,
      showfacets     = false,
      decimation     = 0.0,
    )
end

Makie.plottype(::Scene{D, T}) where {D, T} =
    SceneViz{<:Tuple{<:Tuple{Scene{D,T}}}}

function Makie.plot!(plot::SceneViz{<:Tuple{Scene{D,T}}}) where {D, T}
    # Retrieve scene
    scene = plot[:scene]

    # Plot scene regions
    regions = @lift collect(values($scene.regions))
    if !isempty(regions[])
        MeshViz.viz!(plot, regions,
            size        = plot[:size],
            color       = plot[:region_color],
            colorscheme = plot[:region_colorscheme],
            alpha       = plot[:alpha],
            facetcolor  = plot[:facetcolor],
            showfacets  = plot[:showfacets],
            decimation  = plot[:decimation],
        )
    end

    # Plot scene obstacles
    geom_obstacles =
        @lift collect(Geometry{D,T}, filter(x -> x isa Geometry, $scene.obstacles))
    if !isempty(geom_obstacles[])
        MeshViz.viz!(plot, geom_obstacles,
        size        = plot[:size],
        color       = plot[:obstacle_color],
        colorscheme = plot[:obstacle_colorscheme],
        alpha       = plot[:alpha],
        facetcolor  = plot[:facetcolor],
        showfacets  = plot[:showfacets],
        decimation  = plot[:decimation],
        )
    end
    # TODO: Figure out how to plot a list of meshes through MeshViz

    # Plot scene objects
    geom_objects =
        @lift collect(Geometry{D,T}, Iterators.filter(x -> x isa Geometry, values($scene.objects)))
    if !isempty(geom_objects[])
        MeshViz.viz!(plot, geom_objects,
            size        = plot[:size],
            color       = plot[:object_color],
            colorscheme = plot[:object_colorscheme],
            alpha       = plot[:alpha],
            facetcolor  = plot[:facetcolor],
            showfacets  = plot[:showfacets],
            decimation  = plot[:decimation],
        )
    end
    # TODO: Figure out how to plot a list of meshes through MeshViz
end
