function backup!(tree, b_idx)
    pomdp = tree.pomdp
    γ = discount(pomdp)
    b = tree.b[b_idx]
    d = tree.admissible_cost[b_idx]
    Qa_upper = tree.Qa_upper[b_idx]
    Qca_lower = tree.Qca_lower[b_idx]

    Qa_lower = tree.Qa_lower[b_idx]
    Qca_upper = tree.Qca_upper[b_idx]

    best_Q̄ = -Inf
    best_Qc_ = copy(tree.C_lower[b_idx])

    best_Q_ = -Inf
    best_Q̄c = copy(tree.C_upper[b_idx])

    Vf_upper = -Inf
    Vf_lower = -Inf

    best_a_idx = 0
    best_a = 0

    for a ∈ actions(pomdp)
        Rba = belief_reward(tree, b, a)
        Rca = belief_costs(tree, b, a)
        ba_idx = tree.b_children[b_idx][a]
        Q̄ = Rba
        Qc_ = copy(Rca)
        Q_ = Rba
        Q̄c = copy(Rca)

        for o ∈ observations(pomdp)
            po = tree.poba[ba_idx][o]
            bp_idx = tree.ba_children[ba_idx][o]
            Vao_upper = tree.V_upper[bp_idx]
            Cao_lower = tree.C_lower[bp_idx]
            Q̄ += γ*po*Vao_upper
            Qc_ += γ*po*Cao_lower

            Vao_lower = tree.V_lower[bp_idx]
            Cao_upper = tree.C_upper[bp_idx]
            Q_ += γ*po*Vao_lower
            Q̄c += γ*po*Cao_upper
        end

        Qa_upper[a] = Q̄
        Qca_lower[a] = Qc_
        Qa_lower[a] = Q_
        Qca_upper[a] = Q̄c
        
        Qf_lower = feasibility_value(Qc_, d)
        if Qf_lower > Vf_lower && Vf_lower < 0.
            best_Q̄ = Q̄
            best_Qc_ = Qc_
            Vf_lower = Qf_lower
        elseif Vf_lower > 0. && Q̄ > best_Q̄ && Qf_lower > 0.
            best_Q̄ = Q̄
            best_Qc_ = Qc_
            Vf_lower = Qf_lower
        end

        
        Qf_upper = feasibility_value(Q̄c, d)
        if Qf_upper > Vf_upper && Vf_upper < 0.
            best_Q_ = Q_
            best_Q̄c = Q̄c
            Vf_upper = Qf_upper
            best_a = a
            best_a_idx = ba_idx
        elseif Vf_upper > 0. && Q_ > best_Q_ && Qf_upper > 0.
            best_Q_ = Q_
            best_Q̄c = Q̄c
            Vf_upper = Qf_upper
            best_a = a
            best_a_idx = ba_idx
        end

        compute_action_guarantee!(tree, ba_idx)
    end

    tree.V_upper[b_idx] = best_Q̄
    tree.C_lower[b_idx] = best_Qc_

    tree.V_lower[b_idx] = best_Q_
    tree.C_upper[b_idx] = best_Q̄c

    tree.action[b_idx] = best_a

    compute_belief_guarantee!(tree, b_idx, best_a_idx)
end

function backup!(tree)
    for i ∈ reverse(eachindex(tree.sampled))
        backup!(tree, tree.sampled[i])
    end
end