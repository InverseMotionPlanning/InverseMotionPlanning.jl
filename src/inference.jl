# Export kernels
export drift_kernel_block, drift_kernel_point, drift_kernel_trajectory
export nmc, nmc_multiple_try, nmc_mala, nhmc
export ula_reweight, nmc_reweight
# Export samplers
export mcmc_sampler, rwmh_sampler, mala_sampler, hmc_sampler
export nmc_sampler, nmc_mala_sampler, nhmc_sampler

## MCMC Proposals ##

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

"""
    (new_trace, weight) = ula_reweight(trace, selection::Selection, tau::Real)

Reweighting Langenvin ascent kernel. Instead of accepting or rejecting the 
proposed trace as in MALA, returns an incremental weight along with the trace.
"""
function ula_reweight(
    trace, selection::Selection, tau::Real;
    check=false, observations=EmptyChoiceMap()
)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    std = sqrt(2 * tau)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    # forward proposal
    (_, values_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    forward_mu = values + tau * gradient
    forward_score = 0.
    proposed_values = Vector{Float64}(undef, length(values))
    for i=1:length(values)
        proposed_values[i] = random(Gen.normal, forward_mu[i], std)
        forward_score += logpdf(Gen.normal, proposed_values[i], forward_mu[i], std)
    end

    # evaluate model weight
    constraints = from_array(values_trie, proposed_values)
    (new_trace, up_weight, _, discard) = update(trace,
        args, argdiffs, constraints)
    check && Gen.check_observations(get_choices(new_trace), observations)

    # backward proposal
    (_, _, backward_gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    backward_gradient = to_array(backward_gradient_trie, Float64)
    @assert length(backward_gradient) == length(values)
    backward_score = 0.
    backward_mu  = proposed_values + tau * backward_gradient
    for i=1:length(values)
        backward_score += logpdf(Gen.normal, values[i], backward_mu[i], std)
    end

    # compute and return incremental importance weight
    weight = up_weight - forward_score + backward_score
    return new_trace, weight
end

"""
    new_trace, weight = nmc(trace, selection; step_size = 0.1)

Reweighting Newtonian Monte Carlo (NMC) kernel over [`TrajectoryTrace`]s or
hierarchical traces that contain [`TrajectoryTrace`] subtraces. Returns
the new trace and incremental importance weight.
"""
@inline function nmc_reweight(
    trace::Trace, selection = AllSelection();
    step_size::Real=0.1, target_init=nothing, weight_init=1.0
)
    # Sample proposals for each trajectory subtrace
    fwd_weight = 0.0
    new_choices = nothing
    subtrace_iter = subtrace_selections(trace, selection, TrajectoryTrace)
    for (addr, subtr, subsel) in subtrace_iter
        # Convert sub-selection to trajectory indices
        idxs = selected_idxs(subtr, subsel)

        # Extract values, gradients, and Hessian
        values, g, H = _nmc_extract_values_grads_hessian(subtr, idxs)
        # Decide whether to target initial proposal distribution
        if target_init == :forward # Adjust to match initial proposal
            start, stop = subtr.args.start, subtr.args.stop
            n_points = subtr.args.n_points
            delta = norm(stop .- start) / (n_points - 1)
            # Compute initial proposal distribution
            init_mean = reduce(hcat, LinRange(start, stop, n_points))
            init_mean = vec(init_mean[:, idxs])
            init_std = delta / (2 * weight_init)
            # Compute the product of the Gaussians
            orig_inv_cov = -H ./ (2 * step_size)
            cov = inv(orig_inv_cov + I / (init_std)^2)
            mu = cov * orig_inv_cov * values + 0.5 .* cov * g +
                 (cov * (I / (init_std)^2)) * init_mean
        else  # Compute Newton-Raphson step
            inv_H = inv(H)
            step = (inv_H * g) .* step_size
            mu = values .- step
            cov = -inv_H * (2 * step_size)
        end
        # Sample from multi-variate Gaussian centered at updated location
        new_values = mvnormal(mu, cov)
        fwd_weight += logpdf(mvnormal, new_values, mu, cov)

        # Fill choicemap with new values
        D = embeddim(subtr)
        if subsel isa AllSelection
            subchoices = TrajectoryChoiceMap(reshape(new_values, D, :))
        elseif subsel isa HierarchicalSelection
            new_trajectory = subtr.trajectory[:, 2:end-1]
            new_trajectory[:, idxs .- 1] = reshape(new_values, D, :)
            subchoices = TrajectoryChoiceMap(new_trajectory, idxs.-1)
        end
        if isnothing(addr)
            new_choices = subchoices
        else
            new_choices = isnothing(new_choices) ? choicemap() : new_choices
            set_submap!(new_choices, addr, subchoices)
        end
    end

    # Update trace with new choices
    new_trace, up_weight, _, _ = update(trace, new_choices)

    # Evaluate backward proposal probabilities for each trajectory subtrace
    bwd_weight = 0.0
    subtrace_iter = subtrace_selections(new_trace, selection, TrajectoryTrace)
    for (addr, subtr, subsel) in subtrace_iter
        # Convert sub-selection to trajectory indices
        idxs = selected_idxs(subtr, subsel)

        # Get previous corresponding trajectory subtrace and values
        prev_tr = get_subtrace(trace, addr)
        prev_values = vec(prev_tr.trajectory[:, idxs])

        # Evaluate backward proposal probability
        new_values, g, H = _nmc_extract_values_grads_hessian(subtr, idxs)
        # Decide whether to target initial proposal distribution
        if target_init == :backward # Adjust to match initial proposal
            start, stop = subtr.args.start, subtr.args.stop
            n_points = subtr.args.n_points
            delta = norm(stop .- start) / (n_points - 1)
            # Compute initial proposal distribution
            init_mean = reduce(hcat, LinRange(start, stop, n_points))
            init_mean = vec(init_mean[:, idxs])
            init_std = delta / (2 * weight_init)
            # Compute the product of the Gaussians
            orig_inv_cov = -H ./ (2 * step_size)
            cov = inv(orig_inv_cov + I / (init_std)^2)
            mu = cov * orig_inv_cov * new_values + 0.5 .* cov * g +
                 (cov * (I / (init_std)^2)) * init_mean
        else # Standard NMC proposal
            inv_H = inv(H)
            step = (inv_H * g) .* step_size
            mu = new_values .- step
            cov = -inv_H * (2 * step_size)
        end
        bwd_weight += logpdf(mvnormal, prev_values, mu, cov)
    end

    # Compute incremental importance weight and return
    weight = up_weight - fwd_weight + bwd_weight
    return new_trace, weight
end

"""
    new_trace, accept = nmc(trace, selection; step_size = 0.1)

Newtonian Monte Carlo (NMC) Metropolis-Hastings kernel over [`TrajectoryTrace`]s
or hierarchical traces that contain [`TrajectoryTrace`] subtraces.
"""
function nmc(trace::Trace, selection = AllSelection(); kwargs...)
    # Compute new trace and acceptance ratio
    new_trace, alpha = nmc_reweight(trace, selection; kwargs...)
    # Accept or reject
    if log(rand()) < alpha
        return (new_trace, true)
    else
        return (trace, false)
    end
end

"Multiple-try Newtonian Monte Carlo (NMC) kernel over trajectory traces."
function nmc_multiple_try(
    trace::TrajectoryTrace{D}, selection = AllSelection();
    step_size::Real=0.1, n_tries::Int=5
) where {D}
    # Convert selection to trajectory indices
    if selection isa EmptySelection # Nothing is perturbed
        return trace
    end
    idxs = selected_idxs(trace, selection)

    # Extract values, gradients, and Hessian
    values, g, H = _nmc_extract_values_grads_hessian(trace, idxs)
    # Compute Newton-Raphson step
    inv_H = inv(H)
    step = (inv_H * g) .* step_size

    # Draw multiple samples from NMC proposal distribution
    mu = values .- step
    cov = -inv_H * (2 * step_size)
    fwd_values = [mvnormal(mu, cov) for _ in 1:n_tries]

    # Compute importance weights for each proposed sample
    fwd_weights = map(fwd_values) do val
        trajectory = copy(trace.trajectory)
        trajectory[:, idxs] = reshape(val, D, :)
        prop_score = logpdf(mvnormal, val, mu, cov)
        modeL_score = _trajectory_score(trajectory, trace.args)
        return (modeL_score - prop_score)::Float64
    end

    # Choose one of the proposed samples
    fwd_total_weight = Gen.logsumexp(fwd_weights)
    fwd_probs = exp.(fwd_weights .- fwd_total_weight)
    new_values = fwd_values[categorical(fwd_probs)]

    # Construct updated trace from new values
    if selection isa AllSelection
        new_choices = TrajectoryChoiceMap(reshape(new_values, D, :))
    elseif selection isa HierarchicalSelection
        new_trajectory = trace.trajectory[:, 2:end-1]
        new_trajectory[:, idxs .- 1] = reshape(new_values, D, :)
        new_choices = TrajectoryChoiceMap(new_trajectory, idxs.-1)
    end
    new_trace, _, _, _ = update(trace, new_choices)

    # Draw reference samples from backward proposal
    new_values, g, H = _nmc_extract_values_grads_hessian(new_trace, idxs)
    inv_H = inv(H)
    step = (inv_H * g) .* step_size
    mu = new_values .- step
    cov = -inv_H * (2 * step_size)
    bwd_values = [mvnormal(mu, cov) for _ in 1:n_tries-1]

    # Compute importance weights for each reference sample
    bwd_weights = map(bwd_values) do val
        trajectory = copy(new_trace.trajectory)
        trajectory[:, idxs] = reshape(val, D, :)
        prop_score = logpdf(mvnormal, val, mu, cov)
        modeL_score = _trajectory_score(trajectory, new_trace.args)
        return (modeL_score - prop_score)::Float64
    end
    
    # Add original sample and importance weight
    push!(bwd_values, values)
    push!(bwd_weights, trace.score - logpdf(mvnormal, values, mu, cov))

    # Compute weight ratio and then accept / reject
    bwd_total_weight = Gen.logsumexp(bwd_weights)
    alpha = fwd_total_weight - bwd_total_weight
    if log(rand()) < alpha
        return (new_trace, true)
    else
        return (trace, false)
    end
end

function _nmc_extract_values_grads_hessian(
    trace::TrajectoryTrace{D}, idxs
) where{D}
    # Compute Hessian of smoothness cost (obstacle cost ignored because linear)
    H = hessian(smoothness_cost, trace.trajectory)
    # Multiply by -alpha to get Hessian with respect to score
    H = H .* -trace.args.alpha
    # Restrict Hessian and gradients to selected addresses
    values = vec(trace.trajectory[:, idxs])
    g = vec(trace.trajectory_grads[:, idxs])
    if idxs == 2:length(trace.trajectory)-1
        H = H[D+1:end-D, D+1:end-D]
    else
        h_idxs = reduce(vcat, (D*(i-1)+1:D*i for i in idxs))
        H = H[h_idxs, h_idxs]
    end
    return (values, g, H)
end

# Hybrid NMC + MALA kernel
@pkern function nmc_mala(
    trace, selection = AllSelection();
    nmc_steps::Int=1, mala_steps::Int=1,
    nmc_tries=1, nmc_step_size=1.0, mala_step_size=0.002,
    nmc_step_sizes=Iterators.repeated(nmc_step_size, nmc_steps),
    mala_step_sizes=Iterators.repeated(mala_step_size, mala_steps),
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for step_size in nmc_step_sizes
        trace, _ = nmc_tries == 1 ?
            nmc(trace, selection; step_size) :
            nmc_multiple_try(trace, selection; step_size, n_tries=nmc_tries)
    end
    for tau in mala_step_sizes
        trace, _ = mala(trace, selection, tau)
    end
    accepted = trace !== init_trace
    return trace, accepted
end

# Hybrid NMC + HMC kernel
@pkern function nhmc(
    trace, selection = AllSelection();
    nmc_steps::Int=1, hmc_steps::Int=1,
    nmc_tries=1, nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20,
    nmc_step_sizes=Iterators.repeated(nmc_step_size, nmc_steps),
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for step_size in nmc_step_sizes
        trace, _ = nmc_tries == 1 ?
            nmc(trace, selection; step_size) :
            nmc_multiple_try(trace, selection; step_size, n_tries=nmc_tries)
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
    selection::Selection=AllSelection(), step_size=0.1, n_tries=1, kwargs...
)
    nmc_kernel(trace) = nmc(trace, selection; step_size)
    nmc_multi(trace) = nmc_multiple_try(trace, selection; step_size, n_tries)
    if n_tries == 1
        return mcmc_sampler(trace, n_iters, nmc_kernel; kwargs...)
    else
        return mcmc_sampler(trace, n_iters, nmc_multi; kwargs...)
    end
end

"NMC-MALA hybrid sampler."
function nmc_mala_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(),
    nmc_steps::Int=1, mala_steps::Int=1, nmc_tries::Int=1,
    nmc_step_size=0.2, mala_step_size=0.002,
    nmc_step_sizes=Iterators.repeated(nmc_step_size, nmc_steps),
    mala_step_sizes=Iterators.repeated(mala_step_size, mala_steps),
    kwargs...
)
    nmc_mala_kernel(trace) = nmc_mala(trace, selection; nmc_tries,
                                      nmc_step_sizes, mala_step_sizes)
    return mcmc_sampler(trace, n_iters, nmc_mala_kernel; kwargs...)
end

"Newtonian-Hamiltonian Monte Carlo hybrid sampler."
function nhmc_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(),
    nmc_steps::Int=1, hmc_steps::Int=1, nmc_tries::Int=1,
    nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20,
    nmc_step_sizes=Iterators.repeated(nmc_step_size, nmc_steps),
    kwargs...
)
    nhmc_kernel(trace) = nhmc(trace, selection; nmc_tries, nmc_step_sizes,
                              hmc_steps, hmc_eps, hmc_L)
    return mcmc_sampler(trace, n_iters, nhmc_kernel; kwargs...)
end
