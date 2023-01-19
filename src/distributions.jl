## Useful distributions ##
export LabeledUniform, labeled_uniform, symbol_uniform
export LabeledCategorical, labeled_categorical, symbol_categorical
export RegionUniform, RegionUniform2D, RegionUniform3D
export region_uniform_2D, region_uniform_3D

const ArrayOrTuple{T} = Union{AbstractArray{T},Tuple{Vararg{T}}}

"Labeled uniform distribution over indexable collections."
struct LabeledUniform{T} <: Gen.Distribution{T} end

LabeledUniform() = LabeledUniform{Any}()

(d::LabeledUniform)(args...) = Gen.random(d, args...)

@inline Gen.random(::LabeledUniform{T}, labels::AbstractArray{<:T}) where {T} =
    rand(labels)
@inline Gen.random(::LabeledUniform{T}, labels::AbstractSet{<:T}) where {T} =
    rand(labels)
@inline Gen.random(::LabeledUniform{T}, labels::Tuple{Vararg{<:T}}) where {T} =
    rand(labels)
@inline Gen.logpdf(::LabeledUniform{T}, x::T, labels::ArrayOrTuple{<:T}) where {T} =
    log(sum(x == l for l in labels)) - log(length(labels))
@inline Gen.logpdf(::LabeledUniform{T}, x::T, labels::AbstractSet{<:T}) where {T} =
    x in labels ? -log(length(labels)) : -Inf
    
Gen.logpdf_grad(::LabeledUniform, x, labels) =
    (nothing, nothing)
Gen.has_output_grad(::LabeledUniform) =
    false
Gen.has_argument_grads(::LabeledUniform) =
    (false,)

const labeled_uniform = LabeledUniform{Any}()
const symbol_uniform = LabeledUniform{Symbol}()

"Categorical distribution over an array of labels."
struct LabeledCategorical{T} <: Gen.Distribution{T} end

LabeledCategorical() = LabeledCategorical{Any}()

(d::LabeledCategorical)(args...) = Gen.random(d, args...)

@inline Gen.random(::LabeledCategorical{T}, labels::AbstractArray{<:T}, probs::AbstractArray{<:Real}) where {T} =
    sample(vec(labels), Weights(vec(probs)))
@inline Gen.logpdf(::LabeledCategorical{T}, x::T, labels::AbstractArray{<:T}, probs::AbstractArray{<:Real}) where {T} =
    log(sum((x .== vec(labels)) .* probs))
@inline function Gen.logpdf_grad(::LabeledCategorical{T}, x, labels::AbstractArray{<:T}, probs::AbstractArray{<:Real}) where {T}
    grad = (x .== vec(labels)) .* (1. ./ vec(probs))
    return (nothing, nothing, grad)
end

Gen.has_output_grad(::LabeledCategorical) =
    false
Gen.has_argument_grads(::LabeledCategorical) =
    (false, true)

const labeled_categorical = LabeledCategorical{Any}()
const symbol_categorical = LabeledCategorical{Symbol}()    

"Uniform distribution over geometric regions."
struct RegionUniform{D} <: Gen.Distribution{Vector{Float64}} end

RegionUniform2D() = RegionUniform{2}()
RegionUniform3D() = RegionUniform{3}()

const region_uniform_2D = RegionUniform2D()
const region_uniform_3D = RegionUniform3D()

(d::RegionUniform)(args...) = Gen.random(d, args...)

@inline Gen.random(::RegionUniform{D}, region::Box{D}) where {D} =
    [uniform(region.min.coords[i], region.max.coords[i]) for i in 1:D]
@inline Gen.random(d::RegionUniform{D}, region::Geometry{D}) where {D} =
    Gen.random(d, simplexify(region))
@inline Gen.random(::RegionUniform{D}, region::Domain{D}) where {D} =
    sample(region, HomogeneousSampling(1)) |> first |> coordinates |> Vector
@inline Gen.logpdf(::RegionUniform{D}, x::AbstractVector{<:Real}, region::Geometry{D}) where {D} =
    x in region ? - log(measure(region)) : -Inf
@inline Gen.logpdf(::RegionUniform{D}, x::AbstractVector{<:Real}, region::Domain{D}) where {D} =
    x in region ? - log(measure(region)) : -Inf

Gen.logpdf_grad(::RegionUniform{D}, x::AbstractVector{<:Real}, region::Union{Geometry{D}, Domain{D}}) where {D} =
    (zero(x), nothing)
Gen.has_output_grad(::RegionUniform) =
    true
Gen.has_argument_grads(::RegionUniform) =
    (false,)
