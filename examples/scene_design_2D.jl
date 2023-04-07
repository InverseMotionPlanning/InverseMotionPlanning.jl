# Import libraries and dependencies
using InverseTAMP
using LinearAlgebra
using Meshes
using Gen, GenParticleFilters

# Construct scene
scene = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Box(Point(1, 1), Point(2, 2))
        Box(Point(3, 3), Point(4, 4))
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)

# Visualize scene
figure, axis, plot = sceneviz(scene)

# Generate test trajectory
start = [0.5, 0.5] # Start location
stop = [5.5, 5.5] # Stop location
n_points = 21 # Number of points in trajectory
d_safe = 0.1 # Safe distance
obs_mult = 1.0 # Obstacle multiplier
alpha = 20.0 # Rationality parameter
test_args = (n_points, start, stop, scene, d_safe, obs_mult, alpha)
test_tr = sample_trajectory(2, test_args, verbose=true, n_optim_iters=1)

# Visualize test trajectory
callback = PlotCallback()
callback(test_tr, true)
