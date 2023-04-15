import GenParticleFilters: softmax
export closest_goal, laplace_imp

# Closest goal heuristic cost
function closest_goal(
    obs_trajectory::AbstractArray, 
    scene::Scene,
    alpha::Real
)
    # Get the latest point in the trajectory
    latest_point = obs_trajectory[:, end]

    # Compute the minimum distance to each region
    min_dists = [min_dist(latest_point, r) for r in scene.regions]

    # Compute the probability of reaching each region
    goal_probs = softmax(-alpha*min_dists)

    return goal_probs
end


# Laplace approximation
function laplace_imp(
    obs_trajectory::AbstractArray,
    scene::Scene,
    start::AbstractVector,
    n_points::Int,
    d_safe::Real,
    obs_mult::Real,
    alpha::Real
)
    goal_probs = Float64[]
    for region in scene.regions
        goal = region.center  # assuming the goal is the center of the region

        # Generate the optimal completion of the snippet to the goal
        optimal_completion_args = (goal, n_points, d_safe, obs_mult, alpha)
        optimal_completion = sample_trajectory(length(start), optimal_completion_args, return_best=true)

        # Generate the optimal trajectory from start to goal
        optimal_args = (start, goal, n_points, d_safe, obs_mult, alpha)
        optimal = sample_trajectory(length(start), optimal_args, return_best=true)

        # Compute probability of reaching region given partial trajectory
        log_prob = -trajectory_cost(obs_trajectory) - trajectory_cost(optimal_completion) + trajectory_cost(optimal)
        push!(goal_probs, exp(log_prob))
    end

    # Normalize the goal probabilities
    goal_probs_sum = sum(goal_probs)
    goal_probs_normalized = [prob / goal_probs_sum for prob in goal_probs]

    return goal_probs_normalized
end

function run_experiments(
    full_trajectory::AbstractArray,
    scene::Scene,
    alpha::Real
)
    n_timesteps = size(full_trajectory, 2)
    goals_probs = []

    for t in 1:n_timesteps
        # Create a partial trajectory from the start to the current timestep
        part_trajectory = full_trajectory[:, 1:t]

        # Compute the probability distribution over goals for the current timestep
        goals_prob_now= closest_reg_heuristic(part_trajectory, scene, alpha)

        # Append the probability distribution to the list of all distributions
        push!(goals_probs, goals_prob_now)
    end

    return goals_probs
end