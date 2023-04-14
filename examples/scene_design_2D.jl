# Import libraries and dependencies
using InverseTAMP
using LinearAlgebra
using Meshes
using Gen, GenParticleFilters

# Define obstacles
TOY = [Box(Point(1, 1), Point(2, 2)) # simple two-obstacle toy setup 
       Box(Point(3, 3), Point(4, 4))]

FULL_CNTR = [Triangle((1.5, 1.5), (2.0, 4.0), (4.0, 2.0))] # center Ngon

EMPTY_CNTR = [ Box(Point(1, 1), Point(2, 2)), # big empty-center rhomboid
               Box(Point(3, 3), Point(4, 4)),
               Box(Point(1, 3), Point(2, 4)),
               Box(Point(3, 1), Point(4, 2)),
               Box(Point(0, 2), Point(1, 3)),
               Box(Point(4, 2), Point(5, 3)),
               Box(Point(2, 0), Point(3, 1)),
               Box(Point(2, 4), Point(3, 5))]

MAZE = [Box(Point(0, 1), Point(1, 2)), # varryingly-narrow maze
        Box(Point(0, 2), Point(1, 3)),
        Box(Point(0, 3), Point(1, 4)),
        Box(Point(2, 2), Point(3, 3)),
        Box(Point(3, 1), Point(4, 2)),
        Box(Point(3, 2), Point(4, 3)),
        Box(Point(3, 3), Point(4, 4))]

TUNNEL = [Box(Point(1, -1), Point(2, 0)), # narrow passage
          Box(Point(1, 0), Point(2, 1)),
          Box(Point(1, 1), Point(2, 2)),
          Box(Point(1, 2), Point(2, 3)),
          Box(Point(2, -1), Point(3, 0)),
          Box(Point(2, 0), Point(3, 1)),
          Box(Point(2, 1), Point(3, 2)),
          Box(Point(2, 2), Point(3, 3)),
          Box(Point(3, 5), Point(4, 6)),
          Box(Point(3, 4), Point(4, 5)),
          Box(Point(4, 5), Point(5, 6)),
          Box(Point(4, 4), Point(5, 5))]

#LAPLACE_ADV = [Box(Point(1, 1), Point(2, 2)) # adversary to Laplace method
               #Box(Point(3, 3), Point(4, 4))]

# Construct scene
scene = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),

    obstacles = TUNNEL,

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
