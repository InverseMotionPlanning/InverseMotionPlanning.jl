# Import libraries and dependencies
using InverseTAMP
using LinearAlgebra
using Meshes
using Gen, GenParticleFilters
using DelimitedFiles

import GLMakie:
    GLMakie, Makie, Axis, Figure, Observable, @lift,
    scatter!, events, Mouse, DataAspect, on, off, mouseposition

# Define scenes
PILLARS = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Box(Point(1, 1), Point(2, 2)), 
        Box(Point(3, 3), Point(4, 4))
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)

NGONS = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Ngon((0.5, 1.5), (-0.25, 4.0), (2.0, 3.0)),
        Ngon((3.0, 3.5), (4.0, 4.5), (5.0, 4.0), (5.0, 2.0)),
        Ngon((1.5, -0.25), (2.0, 1.25), (3.0, 1.50), (4.0, 0.25)),
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)

MAZE = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Box(Point(0, 1), Point(1, 4)),
        Box(Point(2, 2), Point(3, 3)),
        Box(Point(3, 1), Point(4, 4))
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)    

TUNNEL = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Box(Point(1, -1), Point(3, 3)), # narrow passage
        Box(Point(3, 4), Point(5, 6))
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)

FOX = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Ngon((0.5, 3.75), (0.5, 5.25), (2.0, 4.5)),
        Ngon((4.5, 3.75), (4.5, 5.25), (3.0, 4.5)),
        Ngon((1.0, 2.5), (2.0, 3.5), (2.0, 1.0)),
        Ngon((4.0, 2.5), (3.0, 3.5), (3.0, 1.0))
    ],
    regions = Dict(
        :A => Box(Point(2, -0.5), Point(3, 0.5)),
        :B => Box(Point(-1, 4), Point(0, 5)),
        :C => Box(Point(5, 4), Point(6, 5))
    )
)

# Visualize scene
scene = FOX
figure, axis, plot = sceneviz(scene)
axis.aspect = DataAspect()
axis.limits = (-1, 6, -1, 6)

# Generate test trajectory
start = [2.5, 4.5] # Start location
stop = [5.5, 4.5] # Stop location
n_points = 21 # Number of points in trajectory
d_safe = 0.2 # Safe distance
obs_mult = 5.0 # Obstacle multiplier
alpha = 20.0 # Rationality parameter
test_args = (n_points, start, stop, scene, d_safe, obs_mult, alpha)

callback = PlotCallback(alpha_as_weight=true, alpha_factor=0.5)
test_tr = sample_trajectory(
    2, test_args, n_replicates=50, verbose=true,
    callback=callback, return_best=true
)

# Visualize test trajectory
callback = PlotCallback()
callback(test_tr, true)

# Generate test trajectories for each scene
SCENES = [PILLARS, NGONS, MAZE, TUNNEL, FOX]
STARTS = [fill([0.5, 0.5], 4); [[2.5, 4.5]]]

for (i, (scene, start)) in enumerate(zip(SCENES, STARTS))
    println("== Scene $i ==")
    i <  4 && continue
    for (goal, region) in scene.regions
        goal != :C && continue
        println("-- Goal $goal --")
        stop = Vector(coordinates(centroid(region)))
        test_args = (n_points, start, stop, scene, d_safe, obs_mult, alpha)
        n_samples = 30
        callback = PlotCallback(alpha_as_weight=false, alpha_factor=0.1)
        traces = sample_trajectory(
            2, test_args, n_samples; n_replicates=50, verbose=true,
            n_mcmc_iters=50,  callback=callback, return_best=true
        )
        println("Visualizing trajectories for selection...")
        count = 0
        for trace in traces
            trajectory = get_retval(trace)
            callback = PlotCallback()
            callback(trace, true)
            sleep(0.1)
            print("Save trajectory? [y/n] ")
            resp = readline()
            if resp == "y"
                count += 1
                println("$count/5 saved")
                writedlm("scene_$(i)_goal_$(goal)_trajectory_$(count).csv", trajectory)
            end
            if count == 5
                break
            end
        end
        println()
    end
    println()
end