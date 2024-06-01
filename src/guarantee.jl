function compute_action_guarantee!(tree, ba_idx::Int)
    min_depth = Inf
    for ba_children ∈ tree.ba_children[ba_idx]
        depth = tree.b_guarantee_depth[ba_children]
        if (depth < min_depth)
            min_depth = depth
        end
    end
    tree.ba_guarantee_depth[ba_idx] = min_depth
end

function compute_belief_guarantee!(tree, b_idx::Int, ba_idx::Int)
    action_guarantee = tree.ba_guarantee_depth[ba_idx]
    if action_guarantee == typemax(Int)
        tree.b_guarantee_depth[b_idx] = action_guarantee
    elseif action_guarantee > 0
        tree.b_guarantee_depth[b_idx] = action_guarantee + 1
    else
        tree.b_guarantee_depth[b_idx] = tree.C_upper[b_idx] < tree.admissible_cost[b_idx] ? 1 : 0
    end
end

function compute_belief_guarantee!(tree, c::Float64, d::Float64)
    guarantee::Int = 0
    if c > d
        guarantee = 0
    elseif has_infinite_guarantee!(tree, c, d)
        guarantee = typemax(Int)
    else
        guarantee = compute_finite_guarantee(tree, c, d)
    end
    return guarantee
end

function has_infinite_guarantee!(tree, c::Float64, d::Float64)
    if tree.max_cost == 0.0 || c == 0.0
        return true
    end
    return tree.max_cost/(1-discount(tree)) < d
end

function compute_finite_guarantee_bruteforce(tree, c::Float64, d::Float64)
    γ = discount(tree)
    depth = 0
    sum = 0
    goal = d/tree.max_cost
    while sum < goal
        depth += 1
        sum += γ^(depth)
    end
    return depth + 1
end

function compute_finite_guarantee(tree, c::Float64, d::Float64)
    #TODO: NOTE: this is incorrect because of cmax and mmc
    γ = discount(tree)
    return floor(log(1 - (d/tree.max_cost)*(1-γ))/log(γ))
end