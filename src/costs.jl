## Costs ##
export obstacle_cost, smoothness_cost, trajectory_cost

function obstacle_cost(p::AbstractVector, scene::Scene, d_safe::Real)
    sd = signed_dist(p, ignore_derivatives(scene))
    return max(d_safe-sd, 0.0)
end

function obstacle_cost(trajectory::AbstractMatrix, scene::Scene, d_safe::Real)
    costs = map(eachcol(trajectory)) do p
        obstacle_cost(p, scene, d_safe)
    end
    return sum(costs)
end

function smoothness_cost(trajectory::AbstractMatrix)
    cost = 0.0
    ix = axes(trajectory, 2)
    for j in ix[2:end]
        diff = trajectory[:, j] - trajectory[:, j-1]
        cost += sum(diff .^ 2)
    end
    return 0.5 * cost
end

function trajectory_cost(trajectory::AbstractMatrix, scene::Scene,
                         d_safe::Real=0.1, obs_mult::Real=1.0)
    return (obs_mult * obstacle_cost(trajectory, scene, d_safe) +
            smoothness_cost(trajectory))
end
