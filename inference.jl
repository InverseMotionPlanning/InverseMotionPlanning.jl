## Proposals ##

"Gaussian drift proposal on each point in a trajectory."
@gen function drift_proposal_point(trace, t::Int, sigma::Real)
    {t} ~ broadcasted_normal(trace.trajectory[:, t+1], sigma)
end

"Gaussian drift proposal on the entire trajectory."
@gen function drift_proposal_trajectory(trace, sigma::Real)
    for t in 1:trace.args.n_points-2
        {t} ~ broadcasted_normal(trace.trajectory[:, t+1], sigma)
    end
end

"Block-wise Gaussian drift proposal on trajectory segments."
@gen function drift_proposal_block(trace, ts, sigma::Real)
    for t in ts
        {t} ~ broadcasted_normal(trace.trajectory[:, t+1], sigma)
    end
end

## MCMC Kernels ##

# Pointwise MH drift kernel.
@kern function drift_kernel_point(trace, sigma)
    n_points = trace.args.n_points
    for t in 1:(n_points-2)
        trace ~ mh(trace, drift_proposal_point, (t, sigma))
    end
end

# Full-trajectory MH drift kernel.
@kern function drift_kernel_trajectory(trace, sigma)
    trace ~ mh(trace, drift_proposal_trajectory, (sigma,))
end

# Block-wise MH drift kernel.
@kern function drift_kernel_block(trace, sigma, block_size)
    n_points = trace.args.n_points
    for t in 1:block_size:(n_points-2)
        ts = t:min(t+block_size-1, n_points-2)
        trace ~ mh(trace, drift_proposal_block, (ts, sigma))
    end
end

"Newtonian Monte Carlo (NMC) kernel over trajectory traces."
function nmc(trace::TrajectoryTrace{D},
             selection = AllSelection(), step_size::Real=1.0) where {D}
    if selection === EmptySelection() return trace end

    # Extract values, gradients, and Hessian
    values, g, H = _nmc_extract_values_grads_hessian(trace, selection)
    # Compute Newton-Raphson step
    inv_H = inv(H)
    step = (inv_H * g) .* step_size
    # Sample from multi-variate Gaussian centered at updated location
    mu = values .- step
    new_values = mvnormal(mu, -inv_H)
    fwd_weight = logpdf(mvnormal, new_values, mu, -inv_H)

    # Construct updated trace from new values
    if selection === AllSelection()
        new_choices = TrajectoryChoiceMap(reshape(new_values, D, :))
    else
        error("Custom indices not supported yet.")
    end
    new_trace, up_weight, _, _ = update(trace, new_choices)

    # Evaluate backward proposal probability
    new_values, g, H = _nmc_extract_values_grads_hessian(trace, selection)
    inv_H = inv(H)
    step = (inv_H * g) .* step_size
    mu = new_values .- step
    bwd_weight = logpdf(mvnormal, values, mu, -inv_H)

    # Perform accept-reject step
    alpha = up_weight - fwd_weight + bwd_weight
    if log(rand()) < alpha
        return (new_trace, true)
    else
        return (trace, false)
    end
end

function _nmc_extract_values_grads_hessian(
    trace::TrajectoryTrace{D}, selection
) where{D}
    # Compute Hessian of smoothness cost (obstacle cost ignored because linear)
    H = hessian(smoothness_cost, trace.trajectory)
    # Multiply by -alpha to get Hessian with respect to score
    H = H .* -trace.args.alpha
    # Restrict Hessian and gradients to selected addresses
    if selection === AllSelection() # Ignore Hessian over endpoints, gradients
        values = vec(trace.trajectory[:, 2:end-1])
        g = vec(trace.trajectory_grads[:, 2:end-1])
        H = H[3:end-2, 3:end-2]
    elseif selection isa HierarchicalSelection
        error("Custom indices not supported yet.")
    end
    return (values, g, H)
end

# Hybrid NMC + MALA kernel
@pkern function nmc_mala(
    trace, selection = AllSelection(), nmc_steps::Int=1, mala_steps::Int=1;
    nmc_step_size=1.0, mala_step_size=0.002, 
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for t in 1:nmc_steps
        trace, _ = nmc(trace, selection, nmc_step_size)
    end
    for t in 1:mala_steps
        trace, _ = mala(trace, selection, mala_step_size)
    end
    accepted = trace !== init_trace
    return trace, accepted
end

# Hybrid NMC + HMC kernel
@pkern function nhmc(
    trace, selection = AllSelection(), nmc_steps::Int=1, hmc_steps::Int=1;
    nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20, 
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for t in 1:nmc_steps
        trace, _ = nmc(trace, selection, nmc_step_size)
    end
    for t in 1:hmc_steps
        trace, _ = hmc(trace, selection; eps=hmc_eps, L=hmc_L)
    end
    accepted = trace !== init_trace
    return trace, accepted
end

## MCMC Samplers ##

"Generic MCMC sampler with configurable kernel."
function mcmc_sampler(
    trace::Trace, n_iters::Int, kernel;
    callback = Returns(nothing)
)
    for i in 1:n_iters
        trace, metadata = kernel(trace)
        callback(trace, metadata)
    end
    return trace
end

"Random-Walk Metropolis Hastings sampler with customizable blocking."
function rwmh_sampler(
    trace::Trace, n_iters::Int;
    block_size = 1, sigma = 0.5, kwargs...
)
    mh_kernel_point(trace) = drift_kernel_point(trace, sigma)
    mh_kernel_all(trace) = drift_kernel_trajectory(trace, sigma)
    mh_kernel_block(trace) = drift_kernel_block(trace, sigma, block_size)
    if block_size == 1
        return mcmc_sampler(trace, n_iters, mh_kernel_point; kwargs...)
    elseif isnothing(block_size) || block_size == :all
        return mcmc_sampler(trace, n_iters, mh_kernel_all; kwargs...)
    else
        return mcmc_sampler(trace, n_iters, mh_kernel_block; kwargs...)
    end
end

"MALA-based sampler."
function mala_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(), tau::Real=0.002, kwargs...
)
    mala_kernel(trace) = mala(trace, selection, tau)
    return mcmc_sampler(trace, n_iters, mala_kernel; kwargs...)
end

"HMC-based sampler."
function hmc_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(), L::Real=20, eps=0.01, kwargs...
)
    hmc_kernel(trace) = hmc(trace, selection; L=L, eps=eps)
    return mcmc_sampler(trace, n_iters, hmc_kernel; kwargs...)
end

"NMC-based sampler."
function nmc_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(), step_size=1.0, kwargs...
)
    nmc_kernel(trace) = nmc(trace, selection, step_size)
    return mcmc_sampler(trace, n_iters, nmc_kernel; kwargs...)
end

"NMC-MALA hybrid sampler."
function nmc_mala_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(),
    nmc_steps::Int=1, mala_steps::Int=1,
    nmc_step_size=0.2, mala_step_size=0.002, kwargs...
)
    nmc_mala_kernel(trace) = nmc_mala(trace, selection, nmc_steps, mala_steps;
                                      nmc_step_size, mala_step_size)
    return mcmc_sampler(trace, n_iters, nmc_mala_kernel; kwargs...)
end

"Newtonian-Hamiltonian Monte Carlo hybrid sampler."
function nhmc_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(),
    nmc_steps::Int=1, hmc_steps::Int=1,
    nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20, kwargs...
)
    nhmc_kernel(trace) = nhmc(trace, selection, nmc_steps, hmc_steps;
                              nmc_step_size, hmc_eps, hmc_L)
    return mcmc_sampler(trace, n_iters, nhmc_kernel; kwargs...)
end

## Callbacks ##

struct PrintCallback
    io::IO
    options::Dict
end

function PrintCallback(io::IO = stdout; kwargs...)
    defaults = Dict{Symbol, Any}(
        :score => true,
        :accepted => false,
        :costs => false,
        :newline => false
    )
    options = merge!(defaults, Dict(kwargs...))
    return PrintCallback(io, options)
end

function (cb::PrintCallback)(trace, accepted)
    if cb.options[:accepted]
        println(cb.io, "Accepted: ", accepted)
    end
    if cb.options[:score]
        println(cb.io, "Score: ", get_score(trace))
    end
    if cb.options[:costs]
        scene, d_safe = trace.args.scene, trace.args.d_safe
        obs_cost = obstacle_cost(trace.trajectory, scene, d_safe)
        smooth_cost = smoothness_cost(trace.trajectory)
        println(cb.io, "Obstacle Cost: ", obs_cost)
        println(cb.io, "Smoothness Cost: ", smooth_cost)
    end
    if cb.options[:newline]
        println(cb.io)
    end
end

mutable struct PlotCallback
    axis::Union{Axis, Axis3, Nothing}
    options::Dict{Symbol, Any}
    observables::Dict{Symbol, Observable}
end

function PlotCallback(axis = nothing, observables = Dict(); kwargs...)
    defaults = Dict{Symbol, Any}(
        :show_scene => true,
        :show_trajectory => true,
        :show_gradients => false,
        :trajectory_color => :red,
        :gradient_color => :blue,
        :gradient_scale => 0.05,
        :sleep => 0.01
    )
    options = merge!(defaults, Dict(kwargs...))
    return PlotCallback(axis, options, observables)
end

function (cb::PlotCallback)(trace, metadata)
    # Extract objects from trace
    trajectory = trace.trajectory
    gradients = trace.trajectory_grads
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, trace)
    else # Update observables then sleep
        if cb.options[:show_trajectory]
            cb.observables[:trajectory][] = trajectory
            if cb.options[:show_gradients]
                cb.observables[:gradients][] = gradients
            end
        end
        if cb.options[:sleep] > 0
            sleep(cb.options[:sleep])
        end
    end
end

function init_plot!(cb::PlotCallback, trace, axis = nothing)
    # Extract objects from trace
    scene = trace.args.scene
    trajectory = trace.trajectory
    gradients = trace.trajectory_grads
    # Construct figure, axis if non-existent
    if isnothing(axis)
        fig = Figure(resolution=(600, 600))
        cb.axis = Axis(fig[1, 1], aspect = 1)
        display(fig)
    else
        cb.axis = axis
    end
    if cb.options[:show_scene]
        scene_obs = Observable(scene)
        plot!(cb.axis, scene_obs, color=:grey)
        cb.observables[:scene] = scene_obs 
    end
    if cb.options[:show_trajectory]
        trajectory_obs = Observable(trajectory)
        color = cb.options[:trajectory_color]
        scatter!(cb.axis, trajectory_obs, color=color)
        lines!(cb.axis, trajectory_obs, color=color)
        cb.observables[:trajectory] = trajectory_obs 
        if cb.options[:show_gradients]
            gradients_obs = Observable(gradients)
            cb.observables[:gradients] = gradients_obs
            scale = cb.options[:gradient_scale] 
            color = cb.options[:gradient_color]
            x = @lift($trajectory_obs[1, :])
            y = @lift($trajectory_obs[2, :])
            u = @lift($gradients_obs[1, :] .* scale)
            v = @lift($gradients_obs[2, :] .* scale)
            arrows!(cb.axis, x, y, u, v, color=color)
        end
    end
    if isnothing(axis)
        display(fig)
    end
end

struct PrintPlotCallback
    print_callback::PrintCallback
    plot_callback::PlotCallback
end

function PrintPlotCallback(
    io::IO = stdout, axis = nothing, observables=Dict();
    kwargs...
)
    print_callback = PrintCallback(io; kwargs...)
    plot_callback = PlotCallback(axis, observables; kwargs...)
    return PrintPlotCallback(print_callback, plot_callback)
end

function (cb::PrintPlotCallback)(trace, accepted)
    cb.print_callback(trace, accepted)
    cb.plot_callback(trace, accepted)
end
