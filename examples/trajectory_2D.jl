## Trajectory sampling example in 2D ##

# Import libraries and dependencies
using InverseTAMP, Gen, Meshes
import GLMakie: GLMakie, Makie, Axis, Figure, Observable, @lift

# Construct scene
b1 = Box(Point(1,1), Point(2, 2))
b2 = Box(Point(3,3), Point(4, 4))
scene = Scene(b1, b2)

# Define trajectory arguments
n_points = 21
start = [0, 0]
stop = [5, 5]

d_safe = 0.1
obs_mult = 1.0
alpha = 20.0

args = (n_points, start, stop, scene, d_safe, obs_mult, alpha)

# Generate initial trajectory trace
tr, w = generate(boltzmann_trajectory_2D, args)

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
tr = nmc_mala_sampler(tr, 100; callback, nmc_steps=1, nmc_step_size=0.2,
                      mala_steps=5, mala_step_size=0.002)
tr = nhmc_sampler(tr, 100; callback, nmc_steps=1, nmc_step_size=0.2,
                  hmc_steps=1, hmc_eps=0.005, hmc_L=10)

## Construct side-by-side comparison animation ##

tr, w = generate(boltzmann_trajectory_2D, args)

fig = Figure(resolution=(1800, 1200))
iter = Observable(0)
iter_string = @lift("Iteration: " * string($iter))

axis = Axis(fig[1, 1], title="Random-Walk Metropolis-Hastings (RWMH)", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
rwmh_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(rwmh_callback, tr, axis)

axis = Axis(fig[1, 2], title="Metropolis-Adjusted Langevin Ascent (MALA)", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
mala_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(mala_callback, tr, axis)

axis = Axis(fig[1, 3], title="Hamiltonian Monte Carlo (HMC)", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
hmc_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(hmc_callback, tr, axis)

axis = Axis(fig[2, 1], title="Newtonian Monte Carlo (NMC)", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
nmc_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(nmc_callback, tr, axis)

axis = Axis(fig[2, 2], title="NMC + MALA", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
nmc_mala_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(nmc_mala_callback, tr, axis)

axis = Axis(fig[2, 3], title="NMC + HMC", aspect=1, titlealign=:left,
            titlesize=22, subtitle=iter_string, subtitlesize=20)
nhmc_callback = PlotCallback(sleep=0.0, show_gradients=false)
init_plot!(nhmc_callback, tr, axis)

traces = fill(tr, 6)
n_iters = 500

Makie.record(fig, "mcmc_comparison.mp4", 1:n_iters, framerate=30) do t
    iter[] = t
    traces[1] = rwmh_sampler(traces[1], 1; callback=rwmh_callback, sigma=0.5, block_size=1)
    traces[2] = mala_sampler(traces[2], 1; callback=mala_callback, tau=0.002)
    traces[3] = hmc_sampler(traces[3], 1; callback=hmc_callback, eps=0.01, L=20)
    traces[4] = nmc_sampler(traces[4], 1; callback=nmc_callback, step_size=0.2)
    traces[5] = nmc_mala_sampler(traces[6], 1; callback=nmc_mala_callback,
                            nmc_steps=1, nmc_step_size=0.2,
                            mala_steps=5, mala_step_size=0.002)
    traces[6] = nhmc_sampler(traces[6], 1; callback=nhmc_callback,
                        nmc_steps=1, nmc_step_size=0.2,
                        hmc_steps=1, hmc_eps=0.005, hmc_L=10)
    sleep(0.001)
end
