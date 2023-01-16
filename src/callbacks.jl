## Callbacks for printing and plotting traces ##
export PrintCallback, PlotCallback, PrintPlotCallback
export init_plot!

"Callback for printing trace information."
struct PrintCallback
    io::IO
    options::Dict
end

function PrintCallback(io::IO = stdout; kwargs...)
    defaults = Dict{Symbol, Any}(
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
        :sleep => 0.01,
        :trajectory_addr => nothing
    )
    options = merge!(defaults, Dict(kwargs...))
    return PlotCallback(axis, options, observables)
end

function (cb::PlotCallback)(trace::Trace, metadata)
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, trace)
        return nothing
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
    # Sleep for the specified time
    if cb.options[:sleep] > 0
        sleep(cb.options[:sleep])
    end
end

function (cb::PlotCallback)(pf_state::ParticleFilterState)
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, pf_state)
        return nothing
    end
    # Iterate over each trace/particle in particle filter
    traces = get_traces(pf_state)
    norm_weights = get_norm_weights(pf_state)
    for (idx, (trace, weight)) in enumerate(zip(traces, norm_weights))
        # Update alpha value
        name = observable_name("trace_alpha", nothing, idx)
        cb.observables[name][] = weight
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
    # Sleep for the specified time
    if cb.options[:sleep] > 0
        sleep(cb.options[:sleep])
    end
end

function init_plot!(cb::PlotCallback, trace::Trace, axis = nothing)
    # Extract scene from trace
    trace_args = get_args(trace)
    scene = trace_args[findfirst(a -> a isa Scene, trace_args)]
    # Initialize axis and plot scene
    init_plot_scene!(cb, scene, axis)
    # Plot trajectories in trace
    init_plot_trajectory!(cb, trace)
end

function init_plot!(cb::PlotCallback, pf_state::ParticleFilterState, axis = nothing)
    # Extract scene from first trace
    trace_args = get_args(get_traces(pf_state)[1])
    scene = trace_args[findfirst(a -> a isa Scene, trace_args)]
    # Initialize axis and plot scene
    init_plot_scene!(cb, scene, axis)
    # Plot each trace/particle in particle filter
    traces = get_traces(pf_state)
    norm_weights = get_norm_weights(pf_state)
    for (idx, (trace, weight)) in enumerate(zip(traces, norm_weights))
        # Plot trajectories in trace
        init_plot_trajectory!(cb, trace, idx, alpha=weight)
    end
end

function init_plot_scene!(cb::PlotCallback, scene::Scene, axis=nothing)
    # Construct figure, axis if non-existent
    dims = paramdim(scene)
    if isnothing(axis)
        fig = Figure(resolution=(600, 600))
        if dims == 2
            cb.axis = Axis(fig[1, 1], aspect = Makie.DataAspect())
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
    if !cb.options[:show_trajectory] return end
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
        scatter!(cb.axis, trajectory_obs, color=(color, alpha))
        lines!(cb.axis, trajectory_obs, color=(color, alpha))
        # Plot gradients
        if cb.options[:show_gradients]
            gradients_obs = Observable(gradients)
            name = observable_name("gradients", addr, idx)
            cb.observables[name] = gradients_obs
            scale = cb.options[:gradient_scale] 
            color = cb.options[:gradient_color]
            x = @lift($trajectory_obs[1, :])
            y = @lift($trajectory_obs[2, :])
            u = @lift($gradients_obs[1, :] .* scale)
            v = @lift($gradients_obs[2, :] .* scale)
            if dims == 2
                arrows!(cb.axis, x, y, u, v, color=(color, alpha))
            elseif dims == 3
                z = @lift($trajectory_obs[3, :])
                w = @lift($gradients_obs[3, :] .* scale)
                arrows!(cb.axis, x, y, z, u, v, w, color=(color, alpha))
            end
        end
    end
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
