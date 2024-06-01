Base.@kwdef struct RecursivelyConstrainedPBVITree
    pomdp::TabularCPOMDP

    b::Vector{SparseVector{Float64,Int}}        = SparseVector{Float64,Int}[]
    admissible_cost::Vector{Vector{Float64}}    = Vector{Float64}[]
    b_children::Vector{UnitRange{Int}}          = UnitRange{Int}[]
    Vs_upper::Vector{Float64}
    Cs_lower::Matrix{Float64}
    V_upper::Vector{Float64}                    = Float64[]
    V_upper_sawtooth::Vector{Float64}           = Float64[]
    V_lower::Vector{Float64}                    = Float64[]
    Qa_upper::Vector{Vector{Float64}}           = Vector{Float64}[]
    Qa_lower::Vector{Vector{Float64}}           = Vector{Float64}[]

    C_upper::Vector{Vector{Float64}}            = Vector{Float64}[]
    C_lower::Vector{Vector{Float64}}            = Vector{Float64}[]
    C_lower_sawtooth::Vector{Vector{Float64}}   = Vector{Float64}[]
    Qca_upper::Vector{Vector{Vector{Float64}}}  = Vector{Vector{Float64}}[]
    Qca_lower::Vector{Vector{Vector{Float64}}}  = Vector{Vector{Float64}}[]

    ba_children::Vector{UnitRange{Int}}         = UnitRange{Int}[]
    poba::Vector{Vector{Float64}}               = Vector{Float64}[]

    is_terminal::BitVector                      = BitVector()

    sampled::Vector{Int}                        = Int[]
    b_pruned::BitVector                         = BitVector()
    ba_pruned::BitVector                        = BitVector()
    real::Vector{Int}                           = Int[]
    is_real::BitVector                          = BitVector()
    cache::TreeCache

    Γ::Vector{AlphaVec}                         = AlphaVec[]

    b_guarantee_depth::Vector{Int}              = Int[]
    ba_guarantee_depth::Vector{Int}             = Int[]

    max_cost::Float64
    action::Vector{Int}                         = Int[]
end

const Tree = RecursivelyConstrainedPBVITree

function RecursivelyConstrainedPBVITree(solver::RecursivelyConstrainedPBVISolver, pomdp::CPOMDP)
    sparse_pomdp = TabularCPOMDP(pomdp)

    # fib gives upper bound reward and lower bound cost
    # blind gives lower bound reward and upper bound cost
    Γ_point = solve(solver.init_upper, sparse_pomdp)
    corner_upper_v = max_upper_val(Γ_point)
    corner_lower_c = min_lower_constraint(Γ_point)

    Γ_alpha = solve(solver.init_lower, sparse_pomdp)

    tree = RecursivelyConstrainedPBVITree(
        pomdp = sparse_pomdp,
        Vs_upper = corner_upper_v,
        Cs_lower = corner_lower_c,
        cache = TreeCache(sparse_pomdp),
        max_cost = max_min_cost(Γ_alpha),
    )
    for α ∈ Γ_alpha
        push!(tree.Γ, α)
    end

    return insert_root!(
        solver,
        tree,
        _initialize_belief(pomdp, initialstate(pomdp))
    )
end

function insert_root!(solver, tree::RecursivelyConstrainedPBVITree, b)
    pomdp = tree.pomdp

    d = ĉ = pomdp.constraints
    a, V̲, C̄ =  lower_feasibility(tree, b, d)
    push!(tree.b, b)
    push!(tree.admissible_cost, ĉ)
    push!(tree.b_children, NO_CHILDREN)
    push!(tree.V_upper, init_root_value(tree, b))
    push!(tree.V_upper_sawtooth, init_root_value(tree, b))
    push!(tree.real, 1)
    push!(tree.is_real, true)
    push!(tree.V_lower, V̲)
    push!(tree.Qa_upper, Float64[])
    push!(tree.Qa_lower, Float64[])
    push!(tree.C_lower, init_root_costs(tree, b))
    push!(tree.C_lower_sawtooth, init_root_costs(tree, b))
    push!(tree.C_upper,  C̄)
    push!(tree.Qca_upper, Vector{Float64}[])
    push!(tree.Qca_lower, Vector{Float64}[])
    push!(tree.b_pruned, false)
    push!(tree.is_terminal, is_terminal_belief(pomdp, b))
    push!(tree.action, a)
    push!(tree.b_guarantee_depth, compute_belief_guarantee!(tree, only(C̄), only(d)))
    fill_belief!(tree, 1)
    return tree
end

belief_feasibility(tree, b, d, a) = minimum(d .- transpose(@view(tree.pomdp.C[:,a,:]))*b)

const NO_CHILDREN = 1:0

POMDPs.states(tree::RecursivelyConstrainedPBVITree) = ordered_states(tree)
POMDPTools.ordered_states(tree::RecursivelyConstrainedPBVITree) = states(tree.pomdp)
POMDPs.actions(tree::RecursivelyConstrainedPBVITree) = ordered_actions(tree)
POMDPTools.ordered_actions(tree::RecursivelyConstrainedPBVITree) = actions(tree.pomdp)
POMDPs.observations(tree::RecursivelyConstrainedPBVITree) = ordered_observations(tree)
POMDPTools.ordered_observations(tree::RecursivelyConstrainedPBVITree) = observations(tree.pomdp)
POMDPs.discount(tree::RecursivelyConstrainedPBVITree) = discount(tree.pomdp)

function _initialize_belief(pomdp::POMDP, dist::Any=initialstate(pomdp)) #TODO: done
    ns = length(states(pomdp))
    b = zeros(ns)
    for s in support(dist)
        sidx = stateindex(pomdp, s)
        b[sidx] = pdf(dist, s)
    end
    return b
end

function add_belief!(tree::RecursivelyConstrainedPBVITree, b, ba_idx::Int, o, d) #TODO: this function is good
    push!(tree.b, b)
    push!(tree.admissible_cost, d)
    b_idx = length(tree.b)
    push!(tree.b_children, NO_CHILDREN)
    push!(tree.is_real, false)
    push!(tree.Qa_upper, Vector{Float64}[])
    push!(tree.Qa_lower, Vector{Float64}[])
    push!(tree.Qca_upper, Vector{Vector{Float64}}[])
    push!(tree.Qca_lower, Vector{Vector{Float64}}[])

    terminal = iszero(tree.poba[ba_idx][o]) || is_terminal_belief(tree.pomdp, b)
    push!(tree.is_terminal, terminal)

    V̄ = if terminal
        0.
    else
        upper_value(tree, b)
    end

    C_ = if terminal
        zeros(n_constraints(tree.pomdp)) #needs to be the correct vector length
    else
        lower_costs(tree, b, d)
    end

    F̄ = feasibility_value(C_, b, d)
    #TODO: prune if feasibility value is negative
    a, V_, C̄ = if terminal
        0, 0, zeros(n_constraints(tree.pomdp))
    else
        lower_feasibility(tree, b, d)
    end
    push!(tree.V_upper, V̄)
    push!(tree.V_upper_sawtooth, V̄)
    push!(tree.V_lower, V_)
    push!(tree.C_upper, C̄)
    push!(tree.C_lower, C_) #TODO: check if C_lower is initialized properly for a new belief
    push!(tree.C_lower_sawtooth, C_)
    push!(tree.b_pruned, true)
    push!(tree.action, a)
    push!(tree.b_guarantee_depth, compute_belief_guarantee!(tree, only(C̄), only(d)))

    return b_idx, V_, V̄, C_, C̄
end

function add_action!(tree::RecursivelyConstrainedPBVITree, b_idx::Int, a::Int) #TODO: done
    ba_idx = length(tree.ba_children) + 1
    push!(tree.ba_children, NO_CHILDREN)
    push!(tree.ba_pruned, true)
    push!(tree.ba_guarantee_depth, 0)
    return ba_idx
end

function fill_belief!(tree::RecursivelyConstrainedPBVITree, b_idx::Int) #TODO: done
    if isempty(tree.b_children[b_idx])
        fill_unpopulated!(tree, b_idx)
    end
end

"""
Fill p(o|b,a), V̲(τ(bao)), V̄(τ(bao)) ∀ o,a if not already filled
"""
function fill_unpopulated!(tree::RecursivelyConstrainedPBVITree, b_idx::Int) #TODO: done
    pomdp = tree.pomdp
    d = tree.admissible_cost[b_idx]
    γ = discount(tree)
    A = actions(tree)
    O = observations(tree)
    N_OBS = length(O)
    N_ACT = length(A)
    N_C = n_constraints(tree.pomdp)
    b = tree.b[b_idx]
    n_b = length(tree.b)
    n_ba = length(tree.ba_children)

    Qa_upper = Vector{Float64}(undef, N_ACT)
    Qa_lower = Vector{Float64}(undef, N_ACT)
    Qca_upper = [Vector{Float64}(undef, N_C) for _ in 1:N_ACT]
    Qca_lower = [Vector{Float64}(undef, N_C) for _ in 1:N_ACT]
    b_children = (n_ba+1):(n_ba+N_ACT)

    for a in A
        ba_idx = add_action!(tree, b_idx, a)
        ba_children = (n_b+1):(n_b+N_OBS)
        tree.ba_children[ba_idx] = ba_children

        n_b += N_OBS

        predictor = mul!(tree.cache.pred, pomdp.T[a],b)
        poba = zeros(Float64, N_OBS)
        Rba = belief_reward(tree, b, a)
        Q̄ = Rba
        Q_ = Rba
        Q̄c = belief_costs(tree, b, a)
        Qc_ = copy(Q̄c)
        Qc_best = copy(Q̄c)
        push!(tree.poba, poba)
        for o ∈ O
            # belief update
            bp = predictor .* pomdp.O[a][:,o]
            bp = sparse(bp)
            po = sum(bp)
            if po > 0.
                bp ./= po
                poba[o] = po
            end

            d′ = (1/γ)*(tree.admissible_cost[b_idx] .- belief_costs(tree, b, a))
            bp_idx, V̲, V̄, C_, Ĉ = add_belief!(tree, bp, ba_idx, o, d′)
            
            Q̄ += γ*po*V̄
            Q_ += γ*po*V̲
            Q̄c .+= γ*po*Ĉ
            Qc_ .+= γ*po*C_
        end
        Qa_upper[a] = Q̄
        Qa_lower[a] = Q_
        Qca_upper[a] = Q̄c
        Qca_lower[a] = Qc_

        compute_action_guarantee!(tree, ba_idx)
    end
    tree.b_children[b_idx] = b_children
    tree.Qa_upper[b_idx] = Qa_upper
    tree.Qa_lower[b_idx] = Qa_lower

    tree.Qca_upper[b_idx] = Qca_upper
    tree.Qca_lower[b_idx] = Qca_lower
    
    V̄, C_ =  update_V_upper(tree, b_idx, d)
    V_, C̄, ba_idx = update_V_lower(tree, b_idx, d)
    
    tree.V_upper[b_idx] = V̄
    tree.V_lower[b_idx] = V_
    tree.C_lower[b_idx] = C_ 
    # tree.C_lower_sawtooth[b_idx] = Qca_lower[a_f_upper]
    tree.C_upper[b_idx] = C̄

    compute_belief_guarantee!(tree, b_idx, ba_idx)
end
