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

function (cb::PlotCallback)(trace, metadata)
    # Construct figure, axis and observables if non-existent
    if isnothing(cb.axis)
        init_plot!(cb, trace)
        return nothing
    end
    # Get trajectory subtrace if address is specified
    if !isnothing(get(cb.options, :trajectory_addr, nothing))
        trace = get_subtrace(trace, cb.options[:trajectory_addr])
    end
    # Extract objects from trace
    trajectory = trace.trajectory
    gradients = trace.trajectory_grads
    # Update observables then sleep
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

function init_plot!(cb::PlotCallback, trace, axis = nothing)
    # Get trajectory subtrace if address is specified
    if !isnothing(get(cb.options, :trajectory_addr, nothing))
        trace = get_subtrace(trace, cb.options[:trajectory_addr])
    end
    # Extract objects from trace
    scene = trace.args.scene
    trajectory = trace.trajectory
    gradients = trace.trajectory_grads
    dims = paramdim(scene)
    # Construct figure, axis if non-existent
    if isnothing(axis)
        fig = Figure(resolution=(600, 600))
        if dims == 2
            cb.axis = Axis(fig[1, 1], aspect = 1)
        elseif dims == 3
            cb.axis = Axis3(fig[1, 1])
        else
            error("Plotting in $dims dimensions not supported.")
        end
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
            if dims == 2
                arrows!(cb.axis, x, y, u, v, color=color)
            elseif dims == 3
                z = @lift($trajectory_obs[3, :])
                w = @lift($gradients_obs[3, :] .* scale)
                arrows!(cb.axis, x, y, z, u, v, w, color=color)
            end
        end
    end
    if isnothing(axis)
        display(fig)
    end
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
