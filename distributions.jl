# struct PointNormal{D} <: Distribution{Point{D,Float64}} end

# const point_normal_2D = PointNormal{2}()
# const point_normal_3D = PointNormal{3}()

# (d::PointNormal)(args...) = Gen.random(d, args...)

# Gen.random(d::PointNormal{D}, mu, sigma) where {D} =
#     Point{D,Float64}(Gen.random(Gen.broadcasted_normal, mu, sigma))
# Gen.random(d::PointNormal{D}, mu::Point{D}, sigma) where {D} =
#     Gen.random(d, coordinates(mu), sigma)
# Gen.logpdf(d::PointNormal{D}, x::Point{D}, mu, sigma) where {D} =
#     Gen.logpdf(Gen.broadcasted_normal, coordinates(mu), mu, sigma)
# Gen.logpdf(d::PointNormal{D}, x::Point{D}, mu::Point{D}, sigma) where {D} =
#     Gen.logpdf(d, x, coordinates(mu), sigma)
# Gen.logpdf_grad(d::PointNormal{D}, x::Point{D}, mu, sigma) where {D} =
#     Gen.logpdf_grad(Gen.broadcasted_normal, coordinates(x), mu, sigma)
# Gen.logpdf_grad(d::PointNormal{D}, x::Point{D}, mu::Point{D}, sigma) where {D} =
#     Gen.logpdf_grad(d, x, coordinates(mu), sigma)

# Gen.has_output_grad(d::PointNormal) = true
# Gen.has_argument_grads(d::PointNormal) = (true, true)
