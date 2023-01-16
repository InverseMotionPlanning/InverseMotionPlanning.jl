## Generative function, trace and choicemaps for Boltzmann trajectories ##
export TrajectoryChoiceMap, TrajectoryTrace
export BoltzmannTrajectoryArgs, BoltzmannTrajectoryGF
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

const BoltzmannTrajectoryArgs{N,S} = @NamedTuple begin
    n_points::N
    start::Vector{Float64}
    stop::Vector{Float64}
    scene::S
    d_safe::Float64
    obs_mult::Float64
    alpha::Float64    
end

struct TrajectoryTrace{D} <: Trace
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

## Trajectory GF ##

struct BoltzmannTrajectoryGF{D} <: GenerativeFunction{Matrix{Float64}, TrajectoryTrace{D}} end

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

function _trajectory_grads(trace::TrajectoryTrace)
    return _trajectory_grads(trace.trajectory, trace.args)
end

function Gen.simulate(gen_fn::BoltzmannTrajectoryGF, args::Tuple)
    error("Not implemented.")
end

function Gen.project(trace::TrajectoryTrace, selection::Selection)
    error("Not implemented.")
end

function Gen.project(trace::TrajectoryTrace, selection::EmptySelection)
    0.0
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
    mu, sigma = zeros(n_elements), ones(n_elements)
    delta = norm(stop .- start) / n_points - 1
    noise = broadcasted_normal(mu, sigma) .* delta/10
    trajectory[:, 2:end-1] .+= reshape(noise, D, :)
    prop_weight = logpdf(broadcasted_normal, noise, mu, sigma)
    # Compute trajectory cost, score, and gradients
    score, grads = withgradient(_trajectory_score,
                                trajectory, scene, d_safe, obs_mult, alpha)
    cost = -score / alpha 
    trajectory_grads = grads[1]
    start_grad, stop_grad = trajectory_grads[:, 1], trajectory_grads[:, end]
    arg_grads = (nothing, start_grad, stop_grad, grads[2:end]...)
    arg_grads = BoltzmannTrajectoryArgs{Nothing, Nothing}(arg_grads)
    # Construct trace
    trace = TrajectoryTrace{D}(gen_fn, args, trajectory,
                               arg_grads, trajectory_grads,
                               cost, score)
    # Return trace and weight
    return trace, (score - prop_weight)
end

function Gen.update(
    trace::TrajectoryTrace{D}, args::Tuple,
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
    new_trace = TrajectoryTrace{D}(trace.gen_fn, args, trajectory,
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
    trace::TrajectoryTrace{D}, selection::Selection, 
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

function sample_trajectory(
    n_dims::Int, args::Tuple;
    n_replicates::Int = 20,
    n_mcmc_iters::Int = 10,
    n_optim_iters::Int = 10,
    verbose=false
)
    # Generate initial trajectory traces
    if verbose
        println("Generating $n_replicates initial trajectories...")
    end
    traces = [generate(BoltzmannTrajectoryGF{n_dims}(), args)[1]
              for _ in 1:n_replicates]
    # Diversify initial trajectories via NMC reweigting steps
    if verbose
        println("Diversifying initial trajectories via NMC proposals...")
    end
    for i in 1:n_replicates
        step_size = rand([0.1, 0.2, 0.4, 0.8])
        traces[i], _ = nmc_reweight(traces[i], AllSelection(); step_size)
    end
    # Run stochastic optimization via MCMC
    if verbose
        println("Running $n_mcmc_iters iterations of MCMC (NMC + MALA)...")
    end
    for i in 1:n_replicates
        traces[i] = nmc_mala_sampler(traces[i], n_mcmc_iters;
                                     mala_steps=5, mala_step_size=0.002, 
                                     nmc_steps=1, nmc_step_size=0.5)
    end
    # Optimize each trace via backtracking gradient descent
    if verbose
        println("Running $n_optim_iters iterations of gradient descent...")
    end
    for i in 1:n_replicates
        for k in 1:n_optim_iters
            traces[i] = map_optimize(traces[i], AllSelection())
        end
    end
    # Sample a trace according to its score (unnormalized log probability)
    if verbose
        println("Sampling trajectory from replicates...")
    end
    scores = get_score.(traces)
    total_score = Gen.logsumexp(scores)
    probs = exp.(scores .- total_score)
    trace = traces[categorical(probs)]
    if verbose
        println("Score: ", trace.score)
        println("Cost: ", trace.cost)
    end
    return trace
end
