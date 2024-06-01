function sample_random!(sol, tree)
    empty!(tree.sampled)
    sample_points_random(sol, tree, 1, 0, false)
end

function sample_points_random(sol::RecursivelyConstrainedPBVISolver, tree::RecursivelyConstrainedPBVITree, b_idx::Int, t, leaf_node)
    tree.b_pruned[b_idx] = false
    if !tree.is_real[b_idx]
        tree.is_real[b_idx] = true
        push!(tree.real, b_idx)
    end

    tree.is_terminal[b_idx] && return

    if leaf_node
        return
    end

    leaf_node = false
    if isempty(tree.b_children[b_idx])
        leaf_node = true
    end

    fill_belief!(tree, b_idx)
    γ = discount(tree)

    if t > 105
        return
    else
        a′ = rand(actions(tree.pomdp))
        ba_idx = tree.b_children[b_idx][a′] #line 10
        tree.ba_pruned[ba_idx] = false
        op_idx = rand(observations(tree.pomdp))
        bp_idx = tree.ba_children[ba_idx][op_idx]
        push!(tree.sampled, b_idx)
        sample_points_random(sol, tree, bp_idx, t+1, leaf_node)
    end
end