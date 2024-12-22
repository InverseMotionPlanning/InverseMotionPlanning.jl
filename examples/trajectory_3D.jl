## Trajectory sampling example in 3D ##

# Import libraries and dependencies
using InverseMotionPlanning, Gen, Meshes
import GLMakie: GLMakie, Makie, Axis3, Figure, Observable, @lift

# Construct scene
floor = Box(Point(-1, -1, -1), Point(6, 6, 0))
pillar1 = Box(Point(1, 1, 0), Point(2, 2, 3))
pillar2 = Box(Point(3, 3, 0), Point(4, 4, 5))
scene = Scene(floor, pillar1, pillar2)

# Define trajectory arguments
n_points = 21
start = [0.0, 0.0, 1.0]
stop = [5.0, 5.0, 4.0]

d_safe = 0.1
obs_mult = 1.0
alpha = 20.0

args = (n_points, start, stop, scene, d_safe, obs_mult, alpha)

# Generate initial trajectory trace
tr, w = generate(boltzmann_trajectory_3D, args)

# Setup printing / plotting callbacks
callback = Returns(nothing) # Empty callback
callback = PrintCallback() # Printing callback
callback = PlotCallback(sleep=0.001, show_gradients=false) # Plotting callback
callback = PrintPlotCallback(sleep=0.001, show_gradients=false) # Combined callback

# Run callback on initial trace
callback(tr, true)

# Run MCMC samplers on trajectory trace
tr = rwmh_sampler(tr, 100; callback, sigma=0.5, block_size=1)
tr = mala_sampler(tr, 100; callback, tau=0.002)
tr = hmc_sampler(tr, 100; callback, eps=0.01, L=10)
tr = nmc_sampler(tr, 100; callback, step_size=0.05, n_tries=1)
tr = nmc_mala_sampler(tr, 500; callback, mala_steps=2, mala_step_size=0.002,
    nmc_tries=5, nmc_steps=1, nmc_step_size=0.75)
tr = nhmc_sampler(tr, 100; callback, hmc_steps=1, hmc_eps=0.005, hmc_L=10,
    nmc_tries=5, nmc_steps=1, nmc_step_size=0.75)


# Plot run statistics
n_iters = 100
trace0, w = generate(boltzmann_trajectory_3D, args)

traces = Dict{String,Vector{TrajectoryTrace}}()
traces["rwmh"] = collect_traces(rwmh_sampler, trace0, n_iters; sigma=0.5, block_size=1)
traces["mala"] = collect_traces(mala_sampler, trace0, n_iters; tau=0.002)
traces["hmc"] = collect_traces(hmc_sampler, trace0, n_iters; eps=0.01, L=10)
traces["nmc"] = collect_traces(nmc_sampler, trace0, n_iters; step_size=0.1, n_tries=1)
#traces["nmc_mala"] = collect_traces(nmc_mala_sampler, tr, 100; callback, mala_steps=2,
#    mala_step_size=0.002, nmc_tries=5, nmc_steps=1, nmc_step_size=0.75)
#traces["nhmc"] = collect_traces(nhmc_sampler, tr, 100; callback, hmc_steps=1, hmc_eps=0.005,
#    hmc_L=10, nmc_tries=5, nmc_steps=1, nmc_step_size=0.75)

plot_trajectory_covariance_trace(traces)
plot_cost_distribution(traces)
plot_cost_vs_iteration(traces)


## Record animation of 3D trajectory sampling ##

tr, w = generate(boltzmann_trajectory_3D, args)

fig = Figure(resolution=(800, 800))
axis = Axis3(fig[1, 1], title="NMC + MALA", titlealign=:left, titlesize=22)
callback = PlotCallback(sleep=0.001, show_gradients=false)
init_plot!(callback, tr, axis)

trace = Ref(tr) # Wrap trace in reference so it can be modified by record
n_iters = 500
Makie.record(fig, "mcmc_trajectory_sampling_3D.mp4", 1:n_iters, framerate=30) do t
    trace[] = nmc_mala_sampler(trace[], 1; callback,
        nmc_tries=10, nmc_steps=1, nmc_step_size=1.0,
        mala_steps=5, mala_step_size=0.002)
    sleep(0.0001)
end
