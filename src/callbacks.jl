## Callbacks for printing and plotting traces ##
export CombinedCallback
export PrintCallback, PlotCallback, PrintPlotCallback
export StoreTracesCallback, ParticleFilterStatsCallback
export init_plot!

"Callback that runs each callback in sequence."
struct CombinedCallback{T <: Tuple}
    callbacks::T
end

CombinedCallback(callbacks::Any...) =
    CombinedCallback{typeof(callbacks)}(callbacks)

function (cb::CombinedCallback)(trace::Trace, accepted)
    for callback in cb.callbacks
        callback(trace, accepted)
    end
end

function (cb::CombinedCallback)(pf_state::ParticleFilterState)
    for callback in cb.callbacks
        callback(pf_state)
    end
end

"Callback for printing trace information."
struct PrintCallback
    io::IO
    options::Dict
end

function PrintCallback(io::IO=stdout; kwargs...)
    defaults = Dict{Symbol,Any}(
        :score => true,
        :accepted => false,
        :costs => false,
        :newline => false,
        :trajectory_addr => nothing
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
        # Get trajectory subtrace if address is specified
        if !isnothing(get(cb.options, :trajectory_addr, nothing))
            trace = get_subtrace(trace, cb.options[:trajectory_addr])
        end
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

"Callback for plotting trace information."
mutable struct PlotCallback
    axis::Union{Axis,Axis3,Nothing}
    options::Dict{Symbol,Any}
    observables::Dict{Symbol,Observable}
end

function PlotCallback(axis=nothing, observables=Dict(); kwargs...)
    defaults = Dict{Symbol,Any}(
        :show_scene => true,
        :show_trajectory => true,
        :show_gradients => false,
        :show_observations => true,
        :trajectory_color => :red,
        :observations_color => :black,
        :gradient_color => :blue,
        :gradient_scale => 0.05,
        :weight_as_alpha => true,
        :alpha_factor => 0.5,
        :observations_addr => :observations,
        :show_timestep => true,
        :save_image => false,
        :image_path_prefix => "image_",
        :timestep_arg_index => 1,
        :sleep => 0.01,
    )
    options = merge!(defaults, Dict(kwargs...))
    return PlotCallback(axis, options, observables)
end

function (cb::PlotCallback)(trace::Trace, metadata)
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, trace)
    end
    # Update observations
    if cb.options[:show_observations] && !(trace isa TrajectoryTrace)
        obs_addr = get(cb.options, :observations_addr, :observations)
        observations = trace[obs_addr]
        if observations isa Vector
            observations = reduce(hcat, observations)
        end
        name = observable_name("observations", obs_addr)
        cb.observables[name][] = observations
    end
    # Iterate over trajectory subtraces
    subtrace_iter = subtrace_selections(trace, selectall(), TrajectoryTrace)
    for (addr, tr, _) in subtrace_iter
        # Extract objects from trace
        trajectory = tr.trajectory
        gradients = tr.trajectory_grads
        # Update observables
        if cb.options[:show_trajectory]
            cb.observables[observable_name("trajectory", addr)][] = trajectory
            if cb.options[:show_gradients]
                cb.observables[observable_name("gradients", addr)][] = gradients
            end
        end
    end
    # Update count
    cb.observables[:count][] += 1
    # Save image
    if cb.options[:save_image]
        count = cb.observables[:count][]
        path = cb.options[:image_path_prefix] * string(count) * ".png"
        save(path, cb.axis.parent)
    end
    # Sleep for the specified time
    if cb.options[:sleep] > 0
        sleep(cb.options[:sleep])
    end
end

function (cb::PlotCallback)(pf_state::ParticleFilterState)
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, pf_state)
    end
    # Update observations
    trace = get_traces(pf_state)[1]
    if cb.options[:show_observations] && !(trace isa TrajectoryTrace)
        obs_addr = get(cb.options, :observations_addr, :observations)
        observations = trace[obs_addr]
        if observations isa Vector
            observations = reduce(hcat, observations)
        end
        name = observable_name("observations", obs_addr)
        cb.observables[name][] = observations
    end
    # Iterate over each trace/particle in particle filter
    traces = get_traces(pf_state)
    norm_weights = get_norm_weights(pf_state, true)
    for (idx, (trace, weight)) in enumerate(zip(traces, norm_weights))
        # Update alpha value
        if cb.options[:weight_as_alpha]
            name = observable_name("trace_alpha", nothing, idx)
            cb.observables[name][] = weight^cb.options[:alpha_factor]
        end
        # Iterate over trajectory subtraces
        subtrace_iter = subtrace_selections(trace, selectall(), TrajectoryTrace)
        for (addr, tr, _) in subtrace_iter
            # Extract objects from trace
            trajectory = tr.trajectory
            gradients = tr.trajectory_grads
            # Update observables
            if cb.options[:show_trajectory]
                name = observable_name("trajectory", addr, idx)
                cb.observables[name][] = trajectory
                if cb.options[:show_gradients]
                    observable_name("gradients", addr, idx)
                    cb.observables[name][] = gradients
                end
            end
        end
    end
    # Update count
    cb.observables[:count][] += 1
    # Update timestep
    timestep_arg_idx = get(cb.options, :timestep_arg_idx, 1)
    cb.observables[:timestep][] = get_args(trace)[timestep_arg_idx]
    # Show timestep in caption
    if cb.options[:show_timestep]
        cb.axis.xlabel = "Timestep: " * string(cb.observables[:timestep][])
    end    
    # Save image
    if cb.options[:save_image]
        timestep = cb.observables[:timestep][]
        path = cb.options[:image_path_prefix] * string(timestep) * ".png"
        save(path, cb.axis.parent)
    end    
    # Sleep for the specified time
    if cb.options[:sleep] > 0
        sleep(cb.options[:sleep])
    end
end

function init_plot!(cb::PlotCallback, trace::Trace, axis=nothing)
    # Extract scene from trace
    trace_args = get_args(trace)
    scene = trace_args[findfirst(a -> a isa Scene, trace_args)]
    # Initialize axis and plot scene
    init_plot_scene!(cb, scene, axis)
    # Plot trajectories in trace
    init_plot_trajectory!(cb, trace)
    # Plot observations in trace
    init_plot_observations!(cb, trace)
    # Set iteration / timestep count
    cb.observables[:count] = Observable(0)
end

function init_plot!(cb::PlotCallback, pf_state::ParticleFilterState, axis=nothing)
    # Extract scene from first trace
    trace_args = get_args(get_traces(pf_state)[1])
    scene = trace_args[findfirst(a -> a isa Scene, trace_args)]
    # Initialize axis and plot scene
    init_plot_scene!(cb, scene, axis)
    # Plot each trace/particle in particle filter
    traces = get_traces(pf_state)
    norm_weights = get_norm_weights(pf_state, true)
    for (idx, (trace, weight)) in enumerate(zip(traces, norm_weights))
        # Plot trajectories in trace
        alpha = cb.options[:weight_as_alpha] ? weight : 1.0
        alpha ^= cb.options[:alpha_factor]
        init_plot_trajectory!(cb, trace, idx, alpha=alpha)
    end
    # Plot observations in first trace
    init_plot_observations!(cb, get_traces(pf_state)[1])
    # Set iteration count
    cb.observables[:count] = Observable(0)
    # Set timestep observable
    timestep_arg_idx = get(cb.options, :timestep_arg_idx, 1)
    cb.observables[:timestep] = Observable(trace_args[timestep_arg_idx])
    # Show timestep in caption
    if cb.options[:show_timestep]
        cb.axis.xlabel = "Timestep: " * string(cb.observables[:timestep][])
    end
end

function init_plot_scene!(cb::PlotCallback, scene::Scene, axis=nothing)
    # Construct figure, axis if non-existent
    dims = paramdim(scene)
    if isnothing(axis)
        fig = Figure(resolution=(600, 600))
        if dims == 2
            cb.axis = Axis(fig[1, 1], aspect=Makie.DataAspect())
            # Set limits if specified
            min_coords = Tuple(scene.limits.min.coords)
            max_coords = Tuple(scene.limits.max.coords)
            if !any(min_coords .== -Inf) && !any(max_coords .== Inf)
                x_lims = (min_coords[1], max_coords[1])
                y_lims = (min_coords[2], max_coords[2])
                Makie.limits!(cb.axis, x_lims, y_lims)
            end
        elseif dims == 3
            cb.axis = Axis3(fig[1, 1])
        else
            error("Plotting in $dims dimensions not supported.")
        end
        display(fig)
    else
        cb.axis = axis
    end
    # Plot scene on axis
    if cb.options[:show_scene]
        scene_obs = Observable(scene)
        plot!(cb.axis, scene_obs)
        cb.observables[:scene] = scene_obs
    end
end

function init_plot_trajectory!(
    cb::PlotCallback, trace::Trace, idx=nothing;
    alpha=1.0
)
    cb.options[:show_trajectory] || return
    # Construct observable for transparency / alpha value
    alpha = Observable(alpha)
    cb.observables[observable_name("trace_alpha", nothing, idx)] = alpha
    # Iterate over trajectory subtraces
    subtrace_iter = subtrace_selections(trace, selectall(), TrajectoryTrace)
    for (addr, tr, _) in subtrace_iter
        # Extract objects from trace
        trajectory = tr.trajectory
        gradients = tr.trajectory_grads
        # Plot trajectory
        trajectory_obs = Observable(trajectory)
        name = observable_name("trajectory", addr, idx)
        cb.observables[name] = trajectory_obs
        color = cb.options[:trajectory_color]
        color_obs = @lift (color, $alpha)
        scatter!(cb.axis, trajectory_obs, color=color_obs)
        lines!(cb.axis, trajectory_obs, color=color_obs)
        # Plot gradients
        if cb.options[:show_gradients]
            gradients_obs = Observable(gradients)
            name = observable_name("gradients", addr, idx)
            cb.observables[name] = gradients_obs
            scale = cb.options[:gradient_scale]
            grad_color = cb.options[:gradient_color]
            grad_color_obs = @lift (grad_color, $alpha)
            x = @lift($trajectory_obs[1, :])
            y = @lift($trajectory_obs[2, :])
            u = @lift($gradients_obs[1, :] .* scale)
            v = @lift($gradients_obs[2, :] .* scale)
            if dims == 2
                arrows!(cb.axis, x, y, u, v, color=grad_color_obs)
            elseif dims == 3
                z = @lift($trajectory_obs[3, :])
                w = @lift($gradients_obs[3, :] .* scale)
                arrows!(cb.axis, x, y, z, u, v, w, color=grad_color_obs)
            end
        end
    end
end

function init_plot_observations!(
    cb::PlotCallback, trace::Trace
)
    # Check that observations and plotting flags are present
    trace isa TrajectoryTrace && return
    cb.options[:show_observations] || return
    choices = get_choices(trace)
    obs_addr = get(cb.options, :observations_addr, :observations)
    # Plot observations
    observations = trace[obs_addr]
    if observations isa Vector
        observations = reduce(hcat, observations)
    end
    observations_obs = Observable(observations)
    name = observable_name("observations", obs_addr)
    cb.observables[name] = observations_obs
    color = cb.options[:observations_color]
    scatter!(cb.axis, observations_obs, color=color,
             marker=:xcross, markersize=20)
end

"Construct a name as a `Symbol` for an observable."
function observable_name(name, addr=nothing, idx=nothing)
    name = string(name)
    if !isnothing(addr) # Append address
        name = "$addr => $name"
    end
    if !isnothing(idx) # Append trace/sample index
        name = "$idx => $name"
    end
    return Symbol(name)
end


"Callback for printing and plotting trace information."
struct PrintPlotCallback
    print_callback::PrintCallback
    plot_callback::PlotCallback
end

function PrintPlotCallback(
    io::IO=stdout, axis=nothing, observables=Dict();
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


"Callback to store traces to calculate run statistics."
struct StoreTracesCallback
    traces::Vector{TrajectoryTrace}
end

function StoreTracesCallback()
    return StoreTracesCallback([])
end

function (cb::StoreTracesCallback)(trace, accepted)
    push!(cb.traces, trace)
end

"Callback for logging particle filter statistics."
struct ParticleFilterStatsCallback
    statistics::Dict{Symbol,Any}
    loggers::Dict{Symbol,Any}
end

function ParticleFilterStatsCallback(; kwargs...)
    defaults = Dict{Symbol,Any}(
        # Computes goal probabilities
        :goal_probs => pf -> begin 
            probs = proportionmap(pf, :goal)
            return [probs[k] for k in sort!(collect(keys(probs)))]
        end,
        # Computes MSE between observations and inferred trajectory
        :trajectory_mse => pf -> begin
            mean(pf, :trajectory, :observations) do trajectory, observations
                n_obs = size(observations, 2)
                return sum((trajectory[:, 1:n_obs] .- observations) .^2)
            end
        end,
        # Compute log marginal likelihood
        :log_ml_est => log_ml_estimate
    )
    loggers = merge!(defaults, Dict(kwargs...))
    return ParticleFilterStatsCallback(Dict{Symbol,Any}(), loggers)
end

function (cb::ParticleFilterStatsCallback)(pf_state::ParticleFilterState)
    for (name, logger) in cb.loggers
        history = get(cb.statistics, name, nothing)
        if isnothing(history)
            history = [logger(pf_state)]
            cb.statistics[name] = history
        else
            push!(history, logger(pf_state))
        end
    end
end
