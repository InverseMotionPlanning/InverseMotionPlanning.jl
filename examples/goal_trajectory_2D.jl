# Import libraries and dependencies
using InverseMotionPlanning
using LinearAlgebra
using Meshes
using Gen, GenParticleFilters

import GLMakie: GLMakie, Makie, Axis, Figure, Observable, @lift

# Define trajectory generative function
smc_trajectory_gf = SMCTrajectoryGF{2}(
    n_particles=100,
    kernel_kwargs=Dict(
        :unmc_step_schedule => [0.02],
        :nmc_step_schedule => [0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005],
        :n_ula_iters => 0,
        :n_mala_iters => 1,
        :init_alpha => 1.0
    ),
    fast_update=true
)

# Define observation model
@gen function observation_model((grad)(points::AbstractArray{<:Real}))
    observed = [{t} ~ broadcasted_normal(points[:, t], 0.2)
                for t in axes(points, 2)]
    return reduce(hcat, observed; init=zeros(2, 0))
end

# Define goal trajectory model
@gen function goal_trajectory_model(
    n_obs::Int,
    scene::Scene,
    start::AbstractVector,
    n_points::Int,
    d_safe::Real,
    obs_mult::Real,
    alpha::Real
)
    goal ~ labeled_uniform(keys(scene.regions))
    stop ~ region_uniform_2D(scene.regions[goal])
    trajectory ~ smc_trajectory_gf(n_points, start, stop, scene,
                                   d_safe, obs_mult, alpha)
    observations ~ observation_model(trajectory[:, 1:n_obs])
    return (goal, stop, trajectory, observations)
end

# Construct scene
scene = Scene{2}(
    limits = Box(Point(-1.0, -1.0), Point(6.0, 6.0)),
    obstacles = [
        Box(Point(1, 1), Point(2, 2))
        Box(Point(3, 3), Point(4, 4))
    ],
    regions = Dict(
        :A => Box(Point(5, 5), Point(6, 6)),
        :B => Box(Point(0, 5), Point(1, 6)),
        :C => Box(Point(5, 0), Point(6, 1))
    )
)

# Define model arguments
n_obs = 10
start = [0.5, 0.5]
n_points = 21
d_safe = 0.2
obs_mult = 5.0
alpha = 20.0
args = (n_obs, scene, start, n_points, d_safe, obs_mult, alpha)

# Set up callbacks
callback = Returns(nothing)
callback = PlotCallback(sleep=0.001)
callback = PrintPlotCallback(sleep=0.001, accepted=true)

# Generate initial trace
@time tr, w = generate(goal_trajectory_model, args, choicemap(:goal => :A))
InverseMotionPlanning.get_subtrace(tr, :trajectory).log_z_est

# Run callback on initial trace
callback(tr, true)

# Perform MCMC moves on trajectory
selection = select(:trajectory)
tr = mala_sampler(tr, 100; callback, selection, tau=0.002)
tr = nmc_sampler(tr, 100; callback, selection, step_size=0.05)
tr = nmc_mala_sampler(tr, 100; callback, selection,
                      mala_steps=5, mala_step_size=0.002, 
                      nmc_tries=1, nmc_steps=1, nmc_step_size=0.05)

## Goal inference via particle filtering

# Define forward and backward proposals

"Propose trajectory point at step `t-1` given observation at `t`."
@gen function trajectory_fwd_proposal(trace, t, obs)
    n_points = get_args(trace)[4]
    if 1 < t < n_points # Only propose for non-endpoints
        observed_point = obs[:observations => t]
        {:trajectory => t-1} ~ broadcasted_normal(observed_point, 0.2)
    end
end

"Backward proposal for trajectory point at step `t` given observation at `t`."
@gen function trajectory_bwd_proposal(trace, t, obs)
    n_points = get_args(trace)[4]
    if 1 < t < n_points # Only propose for non-endpoints
        # Guess that point was close to subsequent point
        next_point = trace[:trajectory][:, t+1]
        {:trajectory => t-1} ~ broadcasted_normal(next_point, 0.5)
    end
end

# Define particle filter
function smc_imp(
    scene::Scene,
    obs_trajectory::AbstractMatrix{Float64};
    start = [0.5, 0.5], n_points = 21,
    d_safe = 0.2, obs_mult = 5.0, alpha = 20.0,
    N = 30, K = 10, n_init_iters = 10,
    callback = Returns(nothing)
)
    # Convert observed trajectory to choicemaps
    obs_choices = [choicemap((:observations => t, collect(v)))
                   for (t, v) in enumerate(eachcol(obs_trajectory))]
    # Stratified initialization of particle filter
    args = (0, scene, start, n_points, d_safe, obs_mult, alpha)
    strata = [choicemap((:goal, name)) for name in keys(scene.regions)] 
    pf = pf_initialize(goal_trajectory_model, args, choicemap(), strata, N);    
    # Replicate each particle K times
    pf_replicate!(pf, K)    
    # Create diversity via rejuvenation kernels
    pf_move_accept!(pf, nmc_mala, (select(:trajectory),), n_init_iters;
                    nmc_step_size=0.05);
    callback(pf)
    # Run particle filter
    argdiffs = (UnknownChange(), ntuple(Returns(NoChange()), length(args)-1)...)
    for (t, obs) in enumerate(obs_choices)
        # Update filter state with new observation
        obs = obs_choices[t]
        args = (t, scene, start, n_points, d_safe, obs_mult, alpha)
        # pf_update!(pf, args, argdiffs, obs)
        pf_update!(pf, args, argdiffs, obs,
                   trajectory_fwd_proposal, (t, obs),
                   trajectory_bwd_proposal, (t, obs))
        # Rejuvenate particles
        pf_move_accept!(pf, nmc_mala, (select(:trajectory),), 1;
                        nmc_step_size=0.02)
        callback(pf)
    end
    # Return particle filter
    return pf
end

# Generate test trajectory
test_args = (n_points, start, [5.5, 5.5], scene, d_safe, obs_mult, alpha)
test_tr = sample_trajectory(2, test_args, verbose=true, n_optim_iters=5,
                            return_best=true, callback=PlotCallback())

# Visualize test trajectory
callback = PlotCallback()
callback(test_tr, true)

# Convert test trajectory to observation choicemaps
test_trajectory = test_tr[]
obs_choices = [choicemap((:observations => t, collect(v)))
               for (t, v) in enumerate(eachcol(test_trajectory))]

# Define plotting and logging callback
callback = CombinedCallback(
    ParticleFilterStatsCallback(),
    PlotCallback(sleep=0.001, weight_as_alpha=true, alpha_factor=0.5)
)

# Run particle filter on test trajectory
pf = smc_imp(scene, test_trajectory, callback=callback)

# Get log probability of each goal region
goal_probs = proportionmap(pf, :goal)
display(goal_probs)
