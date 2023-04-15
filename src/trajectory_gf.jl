## Generative function, trace and choicemaps for Boltzmann trajectories ##
export TrajectoryChoiceMap, TrajectoryTrace
export BoltzmannTrajectoryArgs, BoltzmannTrajectoryTrace, BoltzmannTrajectoryGF
export boltzmann_trajectory_2D, boltzmann_trajectory_3D
export sample_trajectory

## Trajectory ChoiceMap ##

struct TrajectoryChoiceMap{M <: AbstractMatrix, I} <: ChoiceMap
    trajectory::M
    idxs::I
end

TrajectoryChoiceMap(trajectory::AbstractMatrix) =
    TrajectoryChoiceMap(trajectory, axes(trajectory, 2))

Gen.has_value(choices::TrajectoryChoiceMap, idx::Int) =
    idx in choices.idxs
Gen.get_value(choices::TrajectoryChoiceMap, idx::Int) =
    idx in choices.idxs ? @inbounds(choices.trajectory[:, idx]) : throw(KeyError(idx))
Gen.get_submap(choices::TrajectoryChoiceMap, addr) =
    EmptyChoiceMap()
Gen.get_values_shallow(choices::TrajectoryChoiceMap) =
    ((i, choices.trajectory[:, i]) for i in choices.idxs)
Gen.get_submaps_shallow(choices::TrajectoryChoiceMap) =
    ()

function Gen.get_selected(
    choices::TrajectoryChoiceMap, selection::HierarchicalSelection
)
    idxs = [i for (i, s) in get_subselections(selection) if s isa AllSelection]
    intersect!(sort!(idxs), choices.idxs)
    return TrajectoryChoiceMap(choices.trajectory, idxs)
end
Gen.get_selected(choices::TrajectoryChoiceMap, ::AllSelection) =
    choices
Gen.get_selected(choices::TrajectoryChoiceMap, ::EmptySelection) =
    EmptyChoiceMap()
    
Base.isempty(choices::TrajectoryChoiceMap) =
    isempty(choices.idxs)

function Gen._fill_array!(
    choices::TrajectoryChoiceMap, arr::Vector{T}, start_idx::Int
) where {T}
    n_elements = length(choices.idxs) * size(choices.trajectory, 1)
    if length(arr) < start_idx + n_elements
        resize!(arr, 2 * (start_idx + n_elements))
    end
    chosen = view(choices.trajectory, :, choices.idxs)
    arr[start_idx:start_idx+n_elements-1] = chosen
    return n_elements
end

function Gen._from_array(
    proto_choices::TrajectoryChoiceMap, arr::Vector{T}, start_idx::Int
) where {T}
    n_elements = length(proto_choices.idxs) * size(proto_choices.trajectory, 1)
    trajectory = zeros(T, size(proto_choices.trajectory))
    trajectory[:, proto_choices.idxs] = arr[start_idx:start_idx+n_elements-1]
    choices = TrajectoryChoiceMap(trajectory, proto_choices.idxs)
    return n_elements, choices
end

## Trajectory Trace ##

"Abstract trajectory trace."
abstract type TrajectoryTrace{D} <: Trace end

Gen.get_gen_fn(trace::TrajectoryTrace) =
    trace.gen_fn
Gen.get_args(trace::TrajectoryTrace) =
    values(trace.args)
Gen.get_retval(trace::TrajectoryTrace) =
    trace.trajectory
Gen.get_score(trace::TrajectoryTrace) =
    trace.score
Gen.get_choices(trace::TrajectoryTrace) =
    TrajectoryChoiceMap(view(trace.trajectory, :, 2:trace.args.n_points-1))

Meshes.embeddim(::TrajectoryTrace{D}) where {D} = D

function selected_idxs(trace::TrajectoryTrace, selection::HierarchicalSelection)
    idxs = sort!(collect(Int, keys(get_subselections(selection)))) .+ 1
    intersect!(sort!(idxs), axes(trace.trajectory, 2))
    return idxs
end
selected_idxs(trace::TrajectoryTrace, ::AllSelection) =
    2:trace.args.n_points-1
selected_idxs(trace::TrajectoryTrace, ::EmptySelection) =
    Int[]

"Arguments to a Boltzmann trajectory distribution."
const BoltzmannTrajectoryArgs{N,S} = @NamedTuple begin
    n_points::N
    start::Vector{Float64}
    stop::Vector{Float64}
    scene::S
    d_safe::Float64
    obs_mult::Float64
    alpha::Float64    
end

"Trace for an unnormalized Boltzmann distribution over trajectories."
struct BoltzmannTrajectoryTrace{D} <: TrajectoryTrace{D}
    "Generative function that generated this trace."
    gen_fn::GenerativeFunction
    "Arguments to generative function."
    args::BoltzmannTrajectoryArgs{Int, Scene{D, Float64}}
    "Returned trajectory, including start and stop."
    trajectory::Matrix{Float64}
    "Gradients of log probability with respect to arguments."
    arg_grads::BoltzmannTrajectoryArgs{Nothing, Nothing}
    "Gradients of log probabliity with respect to each point in trajectory."
    trajectory_grads::Matrix{Float64}
    "Cost of trajectory."
    cost::Float64
    "Unnormalized log probability of trajectory."
    score::Float64
end

## Boltzmann Trajectory GF ##

"Generative function that represents an unnormalized Boltzmann distribution over trajectories."
struct BoltzmannTrajectoryGF{D} <: GenerativeFunction{Matrix{Float64}, BoltzmannTrajectoryTrace{D}} end

const boltzmann_trajectory_2D = BoltzmannTrajectoryGF{2}()
const boltzmann_trajectory_3D = BoltzmannTrajectoryGF{3}()

function _trajectory_score(trajectory::AbstractMatrix, scene::Scene,
                           d_safe::Real, obs_mult::Real, alpha::Real)
    return -alpha * trajectory_cost(trajectory, scene, d_safe, obs_mult)
end

function _trajectory_score(trajectory::AbstractMatrix,
                           args::BoltzmannTrajectoryArgs)
    return _trajectory_score(trajectory, args.scene,
                             args.d_safe, args.obs_mult, args.alpha)
end

function _trajectory_grads(trajectory::AbstractMatrix, scene::Scene,
                           d_safe::Real, obs_mult::Real, alpha::Real)
    f(t) = _trajectory_score(t, scene, d_safe, obs_mult, alpha)
    return Zygote.gradient(f, trajectory)[1]
end

function _trajectory_grads(trajectory::AbstractMatrix,
                           args::BoltzmannTrajectoryArgs)
    return _trajectory_grads(trajectory, args.scene,
                             args.d_safe, args.obs_mult, args.alpha)
end

function _trajectory_grads(trace::BoltzmannTrajectoryTrace)
    return _trajectory_grads(trace.trajectory, trace.args)
end

function Gen.simulate(gen_fn::BoltzmannTrajectoryGF, args::Tuple)
    error("Not implemented.")
end

function Gen.project(trace::BoltzmannTrajectoryTrace, selection::Selection)
    error("Not implemented.")
end

function Gen.project(trace::BoltzmannTrajectoryTrace, ::EmptySelection)
    # Evaluate probability of full trajectory under initial Gaussian proposal
    n_points = trace.args.n_points
    start, stop = trace.args.start, trace.args.stop
    mu = vec(reduce(hcat, LinRange(start, stop, n_points)[2:end-1]))
    delta = norm(stop .- start) / (n_points - 1)
    values = vec(view(trace.trajectory, :, 2:n_points-1))
    prop_weight = logpdf(broadcasted_normal, values, mu, delta / 2)
    return get_score(trace) - prop_weight
end

function Gen.project(trace::BoltzmannTrajectoryTrace, ::AllSelection)
    return get_score(trace)
end

function Gen.generate(
    gen_fn::BoltzmannTrajectoryGF{D}, args::Tuple, constraints::ChoiceMap
) where {D}
    # Extract arguments
    args = BoltzmannTrajectoryArgs{Int, Scene{D, Float64}}(args)
    n_points, start, stop, scene, d_safe, obs_mult, alpha = args
    # Construct straight line trajectory between start and stop
    trajectory = reduce(hcat, LinRange(start, stop, n_points))
    # Perturb intermediate points with Gaussian noise
    n_elements = D * (n_points - 2)
    mu = zeros(n_elements)
    delta = norm(stop .- start) / (n_points - 1)
    noise = broadcasted_normal(mu, delta / 2) 
    prop_weight = logpdf(broadcasted_normal, noise, mu, delta / 2)
    trajectory[:, 2:end-1] .+= reshape(noise, D, :)
    # Compute trajectory cost, score, and gradients
    score, grads = withgradient(_trajectory_score,
                                trajectory, scene, d_safe, obs_mult, alpha)
    cost = -score / alpha 
    trajectory_grads = grads[1]
    start_grad, stop_grad = trajectory_grads[:, 1], trajectory_grads[:, end]
    arg_grads = (nothing, start_grad, stop_grad, grads[2:end]...)
    arg_grads = BoltzmannTrajectoryArgs{Nothing, Nothing}(arg_grads)
    # Construct trace
    trace = BoltzmannTrajectoryTrace{D}(gen_fn, args, trajectory,
                                        arg_grads, trajectory_grads,
                                        cost, score)
    # Return trace and weight
    return trace, (score - prop_weight)
end

function Gen.update(
    trace::BoltzmannTrajectoryTrace{D}, args::Tuple,
    argdiffs::Tuple, constraints::ChoiceMap
) where {D}
    # Extract arguments
    args = BoltzmannTrajectoryArgs{Int, Scene{D, Float64}}(args)
    n_points, start, stop, scene, d_safe, obs_mult, alpha = args
    # Construct new trajectory 
    trajectory = Matrix{Float64}(undef, D, n_points)
    trajectory[:, 1] = start
    trajectory[:, end] = stop
    # Fill in trajectory values from constraints and previous trace
    updated_idxs = Int[]
    prop_weight = 0.0
    for t in 2:n_points-1
        if has_value(constraints, t-1) # Fill in from constraints
            push!(updated_idxs, t-1)         
            trajectory[:, t] = constraints[t-1]
        elseif t < trace.args.n_points # Fill in from previous trace
            trajectory[:, t] = trace.trajectory[:, t]
        end
        # TODO: Fill in new points
    end
    # Compute updated trajectory cost, score, and gradients
    score, grads = withgradient(_trajectory_score,
                                trajectory, scene, d_safe, obs_mult, alpha)
    cost = -score / alpha 
    trajectory_grads = grads[1]
    start_grad, stop_grad = trajectory_grads[:, 1], trajectory_grads[:, end]
    arg_grads = (nothing, start_grad, stop_grad, grads[2:end]...)
    arg_grads = BoltzmannTrajectoryArgs{Nothing, Nothing}(arg_grads)
    # Construct new trace, incremental weight, and discarded choices
    new_trace = BoltzmannTrajectoryTrace{D}(trace.gen_fn, args, trajectory,
                                            arg_grads, trajectory_grads,
                                            cost, score)
    weight = new_trace.score - trace.score
    deleted_idxs = n_points:(trace.args.n_points-1)
    discarded_idxs = append!(updated_idxs, deleted_idxs)
    internal_points = view(trace.trajectory, :, 2:n_points-1)
    discard = TrajectoryChoiceMap(internal_points, discarded_idxs)
    return new_trace, weight, UnknownChange(), discard
end

function Gen.choice_gradients(
    trace::BoltzmannTrajectoryTrace{D}, selection::Selection, 
    retgrad::Union{Nothing, AbstractMatrix}
) where {D}
    # Add retgrad to existing trajectory gradients
    trajectory_grads = trace.trajectory_grads
    if !isnothing(retgrad)
        trajectory_grads = copy(trajectory_grads) .+ retgrad
    end
    # Update gradient with respect to start and end points
    arg_grads = values(trace.arg_grads)
    if !isnothing(retgrad)
        start_grad = trajectory_grads[:, 1]
        stop_grad = trajectory_grads[:, end]
        arg_grads = (nothing, start_grad, stop_grad, arg_grads[4:end]...)
    end
    # Restrict choices to selected addresses
    choice_values = get_selected(get_choices(trace), selection)
    # Restrict trajectory gradients to selected addresses
    n_points = trace.args.n_points
    choice_grads = TrajectoryChoiceMap(view(trajectory_grads, :, 2:n_points-1))
    choice_grads = get_selected(choice_grads, selection)
    return (arg_grads, choice_values, choice_grads)
end

Gen.has_argument_grads(gen_fn::BoltzmannTrajectoryGF) =
    (false, true, true, false, true, true, true)

Gen.accepts_output_grad(gen_fn::BoltzmannTrajectoryGF) =
    true

## Trajectory Sampling and Optimization ##

"""
    trace = sample_trajectory(n_dims::Int, args::Tuple; kwargs...)

Samples an approximately optimal trajectory by performing a combination of 
MCMC and gradient descent on traces drawn from a `BoltzmannTrajectoryGF`.
`n_dims` is the number of dimensions of each trajectory point, and `args`
are the arguments to a `BoltzmannTrajectoryGF`.

# Keyword Arguments
- `n_replicates::Int = 20`: Number of MCMC replicates / chains.
- `n_mcmc_iters::Int = 10`: Number of MCMC iterations.
- `n_optim_iters::Int = 10`: Number of gradient descent iterations.
- `init_alpha::Float64 = 1.0`: Initial value of `alpha` for simulated annealing.
- `return_best::Bool = false`: Flag to return best trace instead of sampling.
- `verbose::Bool = false`: Flag to print more information.
"""
function sample_trajectory(
    n_dims::Int, args::Tuple;
    n_replicates::Int = 20,
    n_mcmc_iters::Int = 30,
    n_optim_iters::Int = 10,
    init_alpha::Float64 = 1.0,
    return_best::Bool = false,
    verbose::Bool=false,
    callback=Returns(nothing)
)
    # Generate initial trajectory traces
    if verbose
        println("Generating $n_replicates initial trajectories...")
    end
    alpha = args[end]
    init_args = (args[1:end-1]..., init_alpha)
    pf = pf_initialize(BoltzmannTrajectoryGF{n_dims}(), init_args,
                       EmptyChoiceMap(), n_replicates)
    callback(pf)
    # Diversify initial trajectories via NMC reweigting steps
    if verbose
        println("Diversifying initial trajectories via NMC proposals...")
    end
    for i in 1:n_replicates
        step_size = rand([0.1, 0.2, 0.4, 0.8])
        pf.traces[i], _ = nmc_reweight(pf.traces[i], AllSelection(); step_size)
        pf.log_weights[i] = get_score(pf.traces[i])
    end
    callback(pf)
    # Run stochastic optimization via MCMC
    if verbose
        println("Running $n_mcmc_iters iterations of MCMC (NMC + MALA)...")
    end
    argdiffs = map((_) -> UnknownChange(), args)
    mult = (alpha / init_alpha) ^ (1 / n_mcmc_iters)
    cur_alpha = init_alpha
    for k in 1:n_mcmc_iters
        # Anneal alpha and update traces with new value
        cur_alpha = k == n_mcmc_iters ? alpha : cur_alpha * mult
        new_args = (args[1:end-1]..., cur_alpha)
        pf_update!(pf, new_args, argdiffs, EmptyChoiceMap())
        # Perturch traces via MCMC
        pf_move_accept!(pf, nmc_mala, (AllSelection(),);
                        mala_steps=5, mala_step_size=0.002, 
                        nmc_steps=2, nmc_step_size=0.1)
        for i in 1:n_replicates
            pf.traces[i] = map_optimize(pf.traces[i], AllSelection())
        end
        pf.log_weights .= get_score.(pf.traces)
        callback(pf)
    end
    # Optimize each trace via backtracking gradient descent
    if verbose
        println("Running $n_optim_iters iterations of gradient descent...")
    end
    for k in 1:n_optim_iters
        for i in 1:n_replicates
            pf.traces[i] = map_optimize(pf.traces[i], AllSelection())
        end
        pf.log_weights .= get_score.(pf.traces)
        callback(pf)
    end
    # Sample or select best trace
    if return_best
        if verbose
            println("Selecting best trajectory among replicates...")
        end
        trace = argmax(get_score, pf.traces)
    else
        if verbose
            println("Sampling trajectory from replicates...")
        end
        trace = rand(pf.traces)
    end
    if verbose
        println("Score: ", trace.score)
        println("Cost: ", trace.cost)
    end
    return trace
end
