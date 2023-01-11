## Trajectory sampling example in 3D ##

# Import libraries and dependencies
using InverseTAMP, Gen, Meshes
import GLMakie: GLMakie, Makie, Axis3, Figure, Observable, @lift

# Construct scene
floor = Box(Point(-1, -1, -1), Point(6, 6, 0))
pillar1 = Box(Point(1, 1, 0), Point(2, 2, 3))
pillar2 = Box(Point(3, 3, 0), Point(4, 4, 5))
scene = Scene(floor, pillar1, pillar2)

# Define trajectory arguments
n_points = 21
start = [0., 0., 1.]
stop = [5., 5., 4.]

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
tr = hmc_sampler(tr, 100; callback, eps=0.01, L=20)
tr = nmc_sampler(tr, 100; callback, step_size=0.2)
tr = nmc_mala_sampler(tr, 100; callback, nmc_steps=1, nmc_step_size=0.05,
                      mala_steps=10, mala_step_size=0.002)
tr = nhmc_sampler(tr, 100; callback, nmc_steps=1, nmc_step_size=0.1,
                  hmc_steps=1, hmc_eps=0.01, hmc_L=20)

## Record animation of 3D trajectory sampling ##
 
tr, w = generate(boltzmann_trajectory_3D, args)

fig = Figure(resolution=(800, 800))
axis = Axis3(fig[1, 1], title="NMC + MALA", titlealign=:left, titlesize=22)
callback = PlotCallback(sleep=0.001, show_gradients=false)
init_plot!(callback, tr, axis)

trace = Ref(tr) # Wrap trace in reference so it can be modified by record
n_iters = 10
Makie.record(fig, "mcmc_trajectory_sampling_3D_test.mp4", 1:n_iters, framerate=30) do t
    trace[] = nmc_mala_sampler(trace[], 1; callback,
                               nmc_steps=1, nmc_step_size=0.2,
                               mala_steps=5, mala_step_size=0.002)
    sleep(0.001)
end
