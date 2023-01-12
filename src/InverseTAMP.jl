module InverseTAMP

using StaticArrays
using LinearAlgebra
using ChainRulesCore
using Meshes
using Gen

import ConvexBodyProximityQueries as CBPQ
import Zygote: withgradient, hessian
import Base: @kwdef

import GLMakie:
    GLMakie, Makie, Axis, Axis3, Figure, Observable, @lift,
    plot!, scatter!, lines!, arrows!
import MeshViz

include("geometry.jl")
include("costs.jl")
include("gen_utils.jl")
include("trajectory_gf.jl")
include("inference.jl")
include("callbacks.jl")

end