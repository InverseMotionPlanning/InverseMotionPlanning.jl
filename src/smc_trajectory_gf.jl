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

@kwdef struct SMCTrajectoryGF{D, F, B} <: GenerativeFunction{Matrix{Float64}, SMCTrajectoryTrace{D}}
    "Number of particles"
    n_particles::Int = 10
    "SMC forward kernel."
    fwd_kernel::F = smc_trajectory_fwd
    "SMC backward kernel."
    bwd_kernel::B = smc_trajectory_bwd
    "Kernel arguments."
    kernel_args::Tuple = ()
    "Kernel keyword arguments."
    kernel_kwargs::Dict{Symbol,Any} = Dict()
    "Flag to use a fast & lower variance (but biased) implementation of update."
    fast_update::Bool = true
end

"Forward kernel for SMC trajectory sampling."
function smc_trajectory_fwd(
    trace::TrajectoryTrace, selection=AllSelection();
    # Number of UNMC kernels
    n_unmc_iters::Int = 2,
    # Unadjusted NMC proposal step size.
    unmc_step_size::Float64 = 0.05,
    # Unadjusted NMC step size schedule
    unmc_step_schedule = fill(unmc_step_size, n_unmc_iters),
    # Number of ULA iteratinos
    n_ula_iters::Int = 0,
    # ULA iteration step size
    ula_step_size::Float64 = 0.002,
    # Unadjusted NMC step size schedule
    ula_step_schedule = fill(ula_step_size, n_ula_iters),
    # Number of NMC Metropolis-Hastings kernels
    n_nmc_iters::Int = 1,
    # NMC proposal step size.
    nmc_step_size::Float64 = 0.05,
    # NMC step size schedule
    nmc_step_schedule = fill(nmc_step_size, n_nmc_iters),
    # Number of MALA kernels
    n_mala_iters::Int = 5,
    # MALA proposal step size.
    mala_step_size::Float64 = 0.002,
    # MALA step size schedule
    mala_step_schedule = fill(mala_step_size, n_mala_iters),
    # How much to adjust reverse kernels towards initial proposal distribution
    weight_init::Float64 = 2.0,
    # Adjustment decay factor after each kernel application
    weight_init_decay::Float64 = 0.5,
)
    weight = 0.0
    # Run UNMC kernels
    for step_size in unmc_step_schedule
        trace, w = nmc_reweight(trace, selection; step_size, weight_init,
                                target_init=:backward)
        weight_init *= weight_init_decay # Gradually adjust reverse kernels
        weight += w
    end
    # Run ULA kernels
    for step_size in ula_step_schedule
        trace, w = ula_reweight(trace, selection, step_size)
        weight += w
    end
    # Run NMC kernels
    for step_size in nmc_step_schedule
        trace, _ = nmc(trace, selection; step_size)
    end
    # Run MALA kernels
    for step_size in mala_step_schedule
        trace, _ = mala(trace, selection, step_size)
    end
    return trace, weight
end

"Backward kernel for SMC trajectory sampling."
function smc_trajectory_bwd(
    trace::TrajectoryTrace, selection=AllSelection();
    # Number of UNMC kernels
    n_unmc_iters::Int = 2,
    # Unadjusted NMC proposal step size.
    unmc_step_size::Float64 = 0.05,
    # Unadjusted NMC step size schedule
    unmc_step_schedule = fill(unmc_step_size, n_unmc_iters),
    # Number of ULA iterations
    n_ula_iters::Int = 0,
    # ULA iteration step size
    ula_step_size::Float64 = 0.002,
    # Unadjusted NMC step size schedule
    ula_step_schedule = fill(ula_step_size, n_ula_iters),
    # Number of NMC Metropolis-Hastings kernels
    n_nmc_iters::Int = 1,
    # NMC proposal step size.
    nmc_step_size::Float64 = 0.05,
    # NMC step size schedule
    nmc_step_schedule = fill(nmc_step_size, n_nmc_iters),
    # Number of MALA kernels
    n_mala_iters::Int = 5,
    # MALA proposal step size.
    mala_step_size::Float64 = 0.002,
    # MALA step size schedule
    mala_step_schedule = fill(mala_step_size, n_mala_iters),
    # How much to adjust reverse kernels towards initial proposal distribution
    weight_init::Float64 = 2.0,
    # Adjustment decay factor after each kernel application
    weight_init_decay::Float64 = 0.5,
)
    weight = 0.0
    # Run reverse MALA kernels
    for step_size in Iterators.reverse(mala_step_schedule)
        trace, _ = mala(trace, selection, step_size)
    end
    # Run reverse NMC kernels
    for step_size in Iterators.reverse(nmc_step_schedule)
        trace, _ = nmc(trace, selection; step_size)
    end
    # Run reverse ULA kernels 
    for step_size in Iterators.reverse(ula_step_schedule)
        trace, w = ula_reweight(trace, selection, step_size)
        weight += w
    end
    # Run reverse UNMC kernels
    n_unmc_iters = length(unmc_step_schedule)
    weight_init *= weight_init_decay ^ (n_unmc_iters - 1)
    for step_size in Iterators.reverse(unmc_step_schedule)
        trace, w = nmc_reweight(trace, selection; step_size, weight_init,
                                target_init=:forward)
        weight_init /= weight_init_decay # Gradually adjust reverse kernels
        weight += w
    end
    return trace, weight
end

SMCTrajectoryGF{D}(args...; kwargs...) where {D} =
    SMCTrajectoryGF{D, typeof(smc_trajectory_fwd), typeof(smc_trajectory_bwd)}(args...; kwargs...)

function Gen.simulate(gen_fn::SMCTrajectoryGF{D}, args::Tuple) where {D}
    # Initialize internal particle filter
    n_particles = gen_fn.n_particles
    target_fn = BoltzmannTrajectoryGF{D}()
    pf = pf_initialize(target_fn, args, EmptyChoiceMap(), n_particles)
    # Apply SMC forward kernel to each trace
    pf_move_reweight!(pf, gen_fn.fwd_kernel, gen_fn.kernel_args;
                      gen_fn.kernel_kwargs...)
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
    # Apply SMC forward kernel to each trace
    pf_move_reweight!(pf, gen_fn.fwd_kernel, gen_fn.kernel_args;
                      gen_fn.kernel_kwargs...)
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
        bwd_trace, bwd_weight =
            gen_fn.bwd_kernel(new_chosen_trace, gen_fn.kernel_args...;
                              gen_fn.kernel_kwargs...)
        bwd_weight += get_score(new_chosen_trace)
        # Evaluate importance weight of trace under initial proposal
        init_weight = project(bwd_trace, EmptySelection())
        # Compute importance weight for newly chosen trace
        bwd_weight -= init_weight
        new_chosen_weight = get_score(new_chosen_trace) - bwd_weight
    end
    # If arguments do not change, retain other particle weights
    if all(isa.(argdiffs, NoChange)) || all(args .== values(trace.args))
        new_log_w_sum = trace.log_z_est + log(gen_fn.n_particles)
        new_log_w_sum = log(exp(new_log_w_sum) - exp(trace.chosen_weight))
    else # Rerun particle filter for other traces if arguments change
        n_particles = gen_fn.n_particles - 1
        target_fn = get_gen_fn(new_chosen_trace)
        pf = pf_initialize(target_fn, args, EmptyChoiceMap(), n_particles)
        pf_move_reweight!(pf, gen_fn.fwd_kernel, gen_fn.kernel_args;
                          gen_fn.kernel_kwargs...)
        new_log_w_sum = logsumexp(pf.log_weights)
    end
    # Update estimate of log normalizing constant and normalized log probability
    new_log_w_sum = logsumexp(new_log_w_sum, new_chosen_weight)
    new_log_z_est = new_log_w_sum - log(gen_fn.n_particles)
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
