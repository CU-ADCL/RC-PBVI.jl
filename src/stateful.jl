mutable struct StatefulPBVIPolicy{P,T,A} <: Policy
    const pomdp::P
    const tree::T
    const actions::A
    const solver
    state::Int
    b::SparseVector{Float64,Int}
    last_a::Int
    steps::Int
    const t_replan::Float64
    const ϵ_replan::Float64
    function StatefulPBVIPolicy(pomdp::P, tree::T, solver::SO; t_replan=0.05, ϵ_replan=solver.epsilon) where {P,T, SO}
        A = ordered_actions(pomdp)
        return new{P,T,typeof(A)}(pomdp, tree, A, solver, 1, first(tree.b), -1, 0, t_replan, ϵ_replan)
    end
end

function reset!(p::StatefulPBVIPolicy)
    p.state = 1
    p.last_a = -1
    p.steps = 0
end

function update_belief!(p::StatefulPBVIPolicy, a_idx::Int, o_idx::Int)
    pomdp = p.tree.pomdp
    pred = predictor(pomdp, p.b, a_idx)
    return p.b = corrector(pomdp, pred, a_idx, o_idx)
end

function min_cost_action(tree, b)
    min_cost = Inf
    value = -Inf
    min_action = 0
    for α ∈ tree.Γ
        c = maximum(dot(α.c_alpha, b))
        v = dot(α.v_alpha, b)
        if c < min_cost
            min_cost = c
            min_action = α.action
            value = v
        elseif c == min_cost && v > value
            min_cost = c
            min_action = α.action
            value = v
        end
    end
    return min_action
end

function POMDPs.action(p::StatefulPBVIPolicy, o)
    
    p.steps += 1
    if p.last_a == -1
        b_idx = p.state
        a_idx = p.tree.action[b_idx]
        a = p.actions[a_idx]
        p.last_a = a_idx
        return a
    end
    
    o_idx = obsindex(p.pomdp, o)
    
    if p.state > 0
        b_idx = p.state
        a_idx = p.last_a
        ba_idx = p.tree.b_children[b_idx][a_idx]
        bp_idx = p.tree.ba_children[ba_idx][o_idx]
        p.state = p.tree.b_children[bp_idx] === NO_CHILDREN ? -1 : bp_idx

        if p.state > 0
            V̲ = p.tree.V_lower[p.state]
            V̄ = p.tree.V_upper[p.state]
            L = V̲
            ΔV = V̄ - V̲
            U = L + p.solver.epsilon*ΔV
            t0 = time()
            while time() - t0 < p.t_replan
                empty!(p.tree.sampled)
                sample_points(p.solver, p.tree, p.state, L, U, 1, p.ϵ_replan)
                backup!(p.tree)
            end
            p.b = p.tree.b[bp_idx]
            a_idx = p.tree.action[bp_idx]
            a = p.actions[a_idx]
            p.last_a = a_idx
            return a
        else
            a_idx = min_cost_action(p.tree, p.b)
            a = p.actions[a_idx]
            update_belief!(p, a_idx, o_idx)
            p.last_a = a_idx
            return a
        end
    else
        a_idx = min_cost_action(p.tree, p.b)
        a = p.actions[a_idx]
        update_belief!(p, a_idx, o_idx)
        p.last_a = a_idx
        return a
    end
end
