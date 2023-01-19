## Generative function, trace and choicemaps for Boltzmann trajectories ##
export SMCTrajectoryTrace, SMCTrajectoryGF
# export smc_trajectory_2D, smc_trajectory_3D

## SMC Trajectory Trace ##

"Trace for an SMC-normalized Boltzmann distribution over trajectories."
struct SMCTrajectoryTrace{D} <: TrajectoryTrace{D}
    "Generative function that generated this trace."
    gen_fn::GenerativeFunction
    "Underlying unnormalized TrajectoryTrace."
    chosen_trace::BoltzmannTrajectoryTrace{D}
    "Final importance weight of chosen trace."
    chosen_weight::Float64
    "Log estimate of the normalizing constant Z."
    log_z_est::Float64
    "Unbiased estimate of the *normalized* log probability of the trace."
    score::Float64
end

function Base.getproperty(trace::SMCTrajectoryTrace, name::Symbol)
    if name in fieldnames(SMCTrajectoryTrace)
        return getfield(trace, name)
    else
        return getproperty(trace.chosen_trace, name)
    end
end

## Trajectory GF ##

@kwdef struct SMCTrajectoryGF{D} <: GenerativeFunction{Matrix{Float64}, SMCTrajectoryTrace{D}}
    "Number of particles"
    n_particles::Int = 10
    "Number of iterations of NMC kernels."
    n_nmc_iters::Int = 2
    "NMC proposal step size."
    nmc_step_size::Float64 = 0.2
    "Number of iterations of ULA kernels."
    n_ula_iters::Int = 2
    "ULA proposal step size."
    ula_step_size::Float64 = 0.002
    "How much to adjust reverse kernels towards initial proposal distribution."
    weight_init::Float64 = 2.0
    "Adjustment decay factor after each kernel application."
    weight_init_decay::Float64 = 0.5
    "Flag to use a fast & lower variance (but biased) implementation of update."
    fast_update::Bool = true
end

function Gen.simulate(gen_fn::SMCTrajectoryGF{D}, args::Tuple) where {D}
    # Initialize internal particle filter
    n_particles = gen_fn.n_particles
    target_fn = BoltzmannTrajectoryGF{D}()
    pf = pf_initialize(target_fn, args, EmptyChoiceMap(), n_particles)
    # Perform NMC and ULA iterations
    selection = AllSelection()
    weight_init = gen_fn.weight_init
    for t in 1:gen_fn.n_nmc_iters
        pf_move_reweight!(pf, nmc_reweight, (selection,);
                          step_size=gen_fn.nmc_step_size,
                          target_init=:backward, weight_init)
        weight_init *= gen_fn.weight_init_decay
    end
    pf_move_reweight!(pf, ula_reweight, (selection, gen_fn.ula_step_size), 
                      gen_fn.n_ula_iters)
    # Estimate log normalizing constant by taking average particle weight
    log_z_est = log_ml_estimate(pf)
    # Sample trace from particle filter according to weight
    chosen_idx = randboltzmann(1:n_particles, pf.log_weights)
    chosen_trace = pf.traces[chosen_idx]
    chosen_weight = pf.log_weights[chosen_idx] 
    # Compute estimate of unnormalized score
    score = get_score(chosen_trace) - log_z_est
    # Construct and return trace
    trace = SMCTrajectoryTrace{D}(gen_fn, chosen_trace, chosen_weight,
                                  log_z_est, score)
    return trace
end

function Gen.project(trace::SMCTrajectoryTrace, selection::Selection)
    error("Not implemented.")
end

function Gen.project(trace::SMCTrajectoryTrace, selection::EmptySelection)
    return (trace.chosen_weight - trace.log_z_est)
end

function Gen.generate(
    gen_fn::SMCTrajectoryGF{D}, args::Tuple, constraints::ChoiceMap
) where {D}
    # TODO: Handle non-empty constraints
    # Initialize internal particle filter
    n_particles = gen_fn.n_particles
    target_fn = BoltzmannTrajectoryGF{D}()
    pf = pf_initialize(target_fn, args, EmptyChoiceMap(), n_particles)
    # Perform NMC and ULA iterations
    selection = AllSelection()
    weight_init = gen_fn.weight_init
    for _ in 1:gen_fn.n_nmc_iters
        pf_move_reweight!(pf, nmc_reweight, (selection,);
                          step_size=gen_fn.nmc_step_size,
                          target_init=:backward, weight_init)
        weight_init *= gen_fn.weight_init_decay
    end
    pf_move_reweight!(pf, ula_reweight, (selection, gen_fn.ula_step_size), 
                      gen_fn.n_ula_iters)
    # Estimate log normalizing constant by taking average particle weight
    log_z_est = log_ml_estimate(pf)
    # Sample trace from particle filter according to weight
    chosen_idx = randboltzmann(1:n_particles, pf.log_weights)
    chosen_trace = pf.traces[chosen_idx]
    chosen_weight = pf.log_weights[chosen_idx] 
    # Compute estimate of unnormalized score
    score = get_score(chosen_trace) - log_z_est
    # Construct trace
    trace = SMCTrajectoryTrace{D}(gen_fn, chosen_trace, chosen_weight,
                                  log_z_est, score)
    # Compute importance weight
    weight = trace.chosen_weight - trace.log_z_est
    return trace, weight
end

function Gen.update(
    trace::SMCTrajectoryTrace{D}, args::Tuple,
    argdiffs::Tuple, constraints::ChoiceMap
) where {D}
    gen_fn = get_gen_fn(trace)
    # Update chosen trace
    old_chosen_trace = trace.chosen_trace
    new_chosen_trace, _, retdiff, discard =
        update(old_chosen_trace, args, argdiffs, constraints)
    if gen_fn.fast_update # Avoid meta-inference 
        new_chosen_weight = trace.chosen_weight -
            get_score(old_chosen_trace) + get_score(new_chosen_trace)
    else # Infer how new chosen trace could have been sampled
        selection = AllSelection()
        bwd_trace, bwd_weight = new_chosen_trace, get_score(new_chosen_trace)
        gen_fn = get_gen_fn(trace)
        for _ in gen_fn.n_ula_iters # Apply backward ULA kernels
            bwd_trace, inc_weight =
                ula_reweight(bwd_trace, selection, gen_fn.ula_step_size)
            bwd_weight += inc_weight
        end
        weight_init = gen_fn.weight_init
        weight_init *= gen_fn.weight_init_decay ^ (gen_fn.n_nmc_iters - 1)
        for _ in gen_fn.n_nmc_iters # Apply backward NMC kernels
            bwd_trace, inc_weight =
                nmc_reweight(bwd_trace, selection; step_size=gen_fn.nmc_step_size,
                            target_init=:forward, weight_init)
            bwd_weight += inc_weight
            weight_init /= gen_fn.weight_init_decay
        end
        # Evaluate probability of sampling trace under initial proposal
        n_points = new_chosen_trace.args.n_points
        start, stop = new_chosen_trace.args.start, new_chosen_trace.args.stop
        delta = norm(stop .- start) / (n_points - 1)
        mu = vec(reduce(hcat, LinRange(start, stop, n_points)[2:end-1]))
        values = vec(bwd_trace.trajectory[:, 2:end-1])
        init_prop_weight = logpdf(broadcasted_normal, values, mu, delta / 2)
        # Compute importance weight for newly chosen trace
        bwd_weight += init_prop_weight - get_score(bwd_trace)
        new_chosen_weight = get_score(new_chosen_trace) - bwd_weight
    end
    # Rerun particle filter for other traces if arguments change
    if all(diff isa NoChange for diff in argdiffs) || all(args .== Base.values(trace.args))
        new_w_sum = exp(trace.log_z_est + log(gen_fn.n_particles)) -
                    exp(trace.chosen_weight)
    else
        n_particles = gen_fn.n_particles - 1
        target_fn = get_gen_fn(new_chosen_trace)
        pf = pf_initialize(target_fn, args, EmptyChoiceMap(), n_particles)
        # Perform NMC and ULA iterations
        selection = AllSelection()
        weight_init = gen_fn.weight_init
        for _ in 1:gen_fn.n_nmc_iters
            pf_move_reweight!(pf, nmc_reweight, (selection,);
                              step_size=gen_fn.nmc_step_size,
                              target_init=:backward, weight_init)
            weight_init *= gen_fn.weight_init_decay
        end
        pf_move_reweight!(pf, ula_reweight, (selection, gen_fn.ula_step_size), 
                          gen_fn.n_ula_iters)
        new_w_sum = exp(logsumexp(pf.log_weights))
    end
    # Update estimate of log normalizing constant and normalized log probability
    new_w_sum += exp(new_chosen_weight)
    new_log_z_est = log(max(new_w_sum, 0.0)) - log(gen_fn.n_particles)
    new_score = get_score(new_chosen_trace) - new_log_z_est
    # Construct updated trace and weight
    new_trace = SMCTrajectoryTrace{D}(gen_fn, new_chosen_trace, new_chosen_weight,
                                      new_log_z_est, new_score)
    weight = new_score - get_score(trace)
    return new_trace, weight, retdiff, discard
end

function Gen.choice_gradients(
    trace::SMCTrajectoryTrace{D}, selection::Selection, 
    retgrad::Union{Nothing, AbstractMatrix}
) where {D}
    # Return gradients of underlying unnormalized trace, ignoring corrections
    return choice_gradients(trace.chosen_trace, selection, retgrad)
end

Gen.has_argument_grads(gen_fn::SMCTrajectoryGF) =
    (false, true, true, false, true, true, true)

Gen.accepts_output_grad(gen_fn::SMCTrajectoryGF) =
    true
