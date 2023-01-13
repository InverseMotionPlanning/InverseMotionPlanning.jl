# Export kernels
export drift_kernel_block, drift_kernel_point, drift_kernel_trajectory
export nmc, nmc_multiple_try, nmc_mala, nhmc
# Export samplers
export mcmc_sampler, rwmh_sampler, mala_sampler, hmc_sampler
export nmc_sampler, nmc_mala_sampler, nhmc_sampler

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
function nmc(
    trace::TrajectoryTrace{D},
    selection = AllSelection();
    step_size::Real=0.1,
    gd_steps::Int=0,
    gd_step_size::Real=0.001
) where {D}
    # Convert selection to trajectory indices
    if selection isa EmptySelection # Nothing is perturbed
        return trace
    elseif selection isa AllSelection
        idxs = 2:trace.args.n_points-1
    elseif selection isa HierarchicalSelection
        idxs = sort!(collect(Int, keys(get_subselections(selection)))) .+ 1
    else 
        error("Unecognized selection type: $(typeof(selection))")
    end

    # Extract values, gradients, and Hessian
    values, g, H = _nmc_extract_values_grads_hessian(trace, idxs)
    # Compute Newton-Raphson step
    inv_H = inv(H)
    step = (inv_H * g) .* step_size
    # Sample from multi-variate Gaussian centered at updated location
    mu = values .- step
    cov = -inv_H * (2 * step_size)
    new_values = mvnormal(mu, cov)
    fwd_weight = logpdf(mvnormal, new_values, mu, cov)

    # Take several gradient steps from new values
    if gd_steps > 0
        args = trace.args
        new_trajectory = copy(trace.trajectory)
        new_trajectory[:, idxs] = reshape(new_values, D, :)
        for _ in 1:gd_step_size
            g = _trajectory_grads(new_trajectory, args)[:, idxs]
            new_trajectory[:, idxs] .+= gd_step_size .* g
        end
        new_values = vec(new_trajectory[:, idxs])
    end

    # Construct updated trace from new values
    if selection isa AllSelection
        new_choices = TrajectoryChoiceMap(reshape(new_values, D, :))
    elseif selection isa HierarchicalSelection
        new_trajectory = copy(trace.trajectory)
        new_trajectory[:, idxs] = reshape(new_values, D, :)
        new_choices = TrajectoryChoiceMap(new_trajectory, idxs.-1)
    end
    new_trace, up_weight, _, _ = update(trace, new_choices)

    # Take several reverse gradient steps from original values
    if gd_steps > 0
        trajectory = copy(trace.trajectory)
        for _ in 1:gd_steps
            g = _trajectory_grads(trajectory, args)[:, idxs]
            trajectory[:, idxs] .-= gd_step_size .* g
        end
        values = vec(trajectory[:, idxs])
    end

    # Evaluate backward proposal probability
    new_values, g, H = _nmc_extract_values_grads_hessian(new_trace, idxs)
    inv_H = inv(H)
    step = (inv_H * g) .* step_size
    mu = new_values .- step
    cov = -inv_H .* (2 * step_size)
    bwd_weight = logpdf(mvnormal, values, mu, cov)

    # Perform accept-reject step
    alpha = up_weight - fwd_weight + bwd_weight
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
    elseif selection isa AllSelection
        idxs = 2:trace.args.n_points-1
    elseif selection isa HierarchicalSelection
        idxs = sort!(collect(Int, keys(get_subselections(selection)))) .+ 1
    else 
        error("Unecognized selection type: $(typeof(selection))")
    end

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
        new_trajectory = copy(trace.trajectory)
        new_trajectory[:, idxs] = reshape(new_values, D, :)
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
    trace, selection = AllSelection(),
    nmc_steps::Int=1, mala_steps::Int=1;
    nmc_tries=1, nmc_step_size=1.0, mala_step_size=0.002, 
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for t in 1:nmc_steps
        trace, _ = nmc_tries == 1 ?
            nmc(trace, selection; step_size=nmc_step_size) :
            nmc_multiple_try(trace, selection; step_size=nmc_step_size)
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
    nmc_tries=1, nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20, 
    check=false, observations = EmptyChoiceMap()
)
    init_trace = trace
    for t in 1:nmc_steps
        trace, _ = nmc_tries == 1 ?
            nmc(trace, selection; step_size=nmc_step_size) :
            nmc_multiple_try(trace, selection; step_size=nmc_step_size)
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
    selection::Selection=AllSelection(), step_size=1.0, n_tries=1,
    gd_steps=0, gd_step_size=0.001, kwargs...
)
    nmc_kernel(trace) = nmc(trace, selection; step_size, gd_steps, gd_step_size)
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
    nmc_step_size=0.2, mala_step_size=0.002, kwargs...
)
    nmc_mala_kernel(trace) = nmc_mala(trace, selection, nmc_steps, mala_steps;
                                      nmc_tries, nmc_step_size, mala_step_size)
    return mcmc_sampler(trace, n_iters, nmc_mala_kernel; kwargs...)
end

"Newtonian-Hamiltonian Monte Carlo hybrid sampler."
function nhmc_sampler(
    trace::Trace, n_iters::Int;
    selection::Selection=AllSelection(),
    nmc_steps::Int=1, hmc_steps::Int=1, nmc_tries::Int=1,
    nmc_step_size=1.0, hmc_eps=0.01, hmc_L=20, kwargs...
)
    nhmc_kernel(trace) = nhmc(trace, selection, nmc_steps, hmc_steps;
                              nmc_tries, nmc_step_size, hmc_eps, hmc_L)
    return mcmc_sampler(trace, n_iters, nhmc_kernel; kwargs...)
end
