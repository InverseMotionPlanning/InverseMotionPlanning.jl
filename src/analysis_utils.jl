export algo_names, collect_traces, plot_trajectory_covariance_trace, plot_cost_distribution, plot_cost_vs_iteration

import LinearAlgebra
import GLMakie

algo_names = Dict(
    "rwmh" => "Random-Walk Metropolis-Hastings (RWMH)",
    "mala" => "Metropolis-adjusted Langevin Algorithm (MALA)",
    "hmc" => "Hamiltonian Monte Carlo (HMC)",
    "nmc" => "Newtonian Monte Carlo (NMC)",
)


function collect_traces(sampler, trace0::Trace, n_iters::Int; kwargs...)
    callback = StoreTracesCallback()
    callback(trace0, true)
    sampler(trace0, n_iters; callback, kwargs...) # Run sampler
    return callback.traces
end


function plot_trajectory_covariance_trace(traces::Dict{String,Vector{TrajectoryTrace}})
    cov_traces = Dict{String,Float64}()
    for (key, trs) in traces
        trajectories = map(tr -> vec(tr.trajectory), trs)
        cov_traces[key] = LinearAlgebra.tr(cov(trajectories))
    end

    GLMakie.barplot(1:length(cov_traces), collect(values(cov_traces)),
        axis=(xticks=(1:length(cov_traces), collect(keys(cov_traces))),
            title="Trace of Trajectory Covariance Matrix", aspect=1, titlealign=:left, titlesize=22),
    )
end


function plot_cost_distribution(traces::Dict{String,Vector{TrajectoryTrace}})
    costs = Dict{String,Vector{Float64}}()
    for (key, trs) in traces
        costs[key] = map(tr -> tr.cost, trs)
    end
    GLMakie.violin(repeat(1:length(costs), inner=101), vcat(values(costs)...),
        axis=(xticks=(1:length(costs), collect(keys(costs))),
            title="Trajectory Cost", aspect=1, titlealign=:left, titlesize=22),
    )
end


function plot_cost_vs_iteration(traces::Dict{String,Vector{TrajectoryTrace}})
    costs = Dict{String,Vector{Float64}}()
    for (key, trs) in traces
        costs[key] = map(tr -> tr.cost, trs)
    end
    fig = GLMakie.Figure(resolution=(1000, 1000))
    axis = GLMakie.Axis(fig[1, 1], title="Trajectory Cost vs Iteration", aspect=1, titlealign=:left, titlesize=22)
    for (key, cs) in costs
        GLMakie.lines!(axis, collect(1:length(cs)), cs, label=algo_names[key])
    end
    GLMakie.axislegend(axis)
end