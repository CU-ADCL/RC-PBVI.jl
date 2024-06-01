init_root_value(tree::RecursivelyConstrainedPBVITree, b::AbstractVector) = dot(tree.Vs_upper, b)

init_root_costs(tree::RecursivelyConstrainedPBVITree, b::AbstractVector) = transpose(tree.Cs_lower)*b

# FIXME: Dedicated method from NativeSARSOP.jl handling v1 as SparseVector
function min_ratio(v1::AbstractVector, v2::AbstractSparseVector)
    min_ratio = Inf
    I,V = v2.nzind, v2.nzval
    for (a,b) ∈ zip(@view(v1[I]),V)
        ratio = a / b
        ratio < min_ratio && (min_ratio = ratio)
    end
    return min_ratio
end

function upper_value(tree::RecursivelyConstrainedPBVITree, b::AbstractVector) # needs to be updated to handle costs
    α_corner = tree.Vs_upper
    V_corner = dot(b, α_corner)
    V_upper = tree.V_upper_sawtooth #V_upper = tree.V_upper
    v̂_min = Inf
    for b_idx ∈ tree.real
        (tree.b_pruned[b_idx] || tree.is_terminal[b_idx]) && continue
        vint = V_upper[b_idx]
        bint = tree.b[b_idx]
        ϕ = min_ratio(b, bint)
        v̂ = V_corner + ϕ * (vint - dot(bint, α_corner))
        v̂ < v̂_min && (v̂_min = v̂)
    end

    return v̂_min
end

function upper_value_real(tree::RecursivelyConstrainedPBVITree, b_idx::Int) # needs to be updated to handle costs
    return tree.V_upper[b_idx], tree.C_lower[b_idx]
end

function lower_value(tree::RecursivelyConstrainedPBVITree, b::AbstractVector)
    MAX_VAL = -Inf
    for α in tree.Γ
        new_val = reward_value(α, b)
        if new_val > MAX_VAL
            MAX_VAL = new_val
        end
    end
    return MAX_VAL
end

function constrained_lower_value(tree::RecursivelyConstrainedPBVITree, b::AbstractVector, d::AbstractVector)
    Vr = -Inf
    Vf = -Inf
    for α in tree.Γ
        Qr = reward_value(α, b)
        Qf = feasibility_value(α, b, d)
        if Qf > Vf && Vf < 0.
            Vf = Qf
            Vr = Qr
        elseif Vf > 0. && Qr > Vr && Qf > 0.
            Vf = Qf
            Vr = Qr
        end
    end
    return Vr
end

function lower_costs(tree::RecursivelyConstrainedPBVITree, b::AbstractVector, d::AbstractVector)
    α_corner = tree.Cs_lower
    C_corner = transpose(α_corner)*b
    C_lower = tree.C_lower_sawtooth #C_lower = tree.C_lower

    ĉ_max = similar(C_corner)
    f̂_max = -Inf
    ĉ = similar(C_corner)
    mul_cache = similar(C_corner)

    for b_idx ∈ tree.real
        (tree.b_pruned[b_idx] || tree.is_terminal[b_idx]) && continue
        c_int = C_lower[b_idx]
        b_int = tree.b[b_idx]
        ϕ = min_ratio(b, b_int)
        ĉ .= C_corner .+ ϕ .* (c_int .- mul!(mul_cache, transpose(α_corner),b_int))
        f̂ = feasibility_value(ĉ,b,d)
        if f̂ > f̂_max
            f̂_max = f̂
            ĉ_max .= ĉ
        end
    end
    return ĉ_max
end

function upper_costs(tree::RecursivelyConstrainedPBVITree, b::AbstractVector, d::AbstractVector)
    Γ = tree.Γ
    c_min = cost_value(first(Γ), b)
    f_min = feasibility_value(c_min, b, d)

    for α ∈ @view(tree.Γ[2:end])
        c = cost_value(α, b)
        f = feasibility_value(α, b, d)
        if f < f_min
            f_min = f
            c_min = c
        end
    end
    return c_min
end

# alpha mats give upper bound on cost, and consequently a lower bound on feasibility
function lower_feasibility(tree::RecursivelyConstrainedPBVITree, b::AbstractVector, d::AbstractVector)
    f_max = -Inf
    new_val = -Inf
    v_max = -Inf
    new_costs = fill(Inf, length(tree.pomdp.constraints))
    best_action = 0
    for α ∈ tree.Γ
        Qc = cost_value(α, b)
        Qf = feasibility_value(Qc, d)
        Qv = reward_value(α, b)
        # f = feasibility_value(α, b, d)
        if (Qf > f_max)
            f_max = Qf
            v_max = Qv
            new_val = reward_value(α, b)
            new_costs = cost_value(α, b)
            best_action = α.action
        elseif (Qf == f_max && Qv > v_max)
            f_max = Qf
            new_val = reward_value(α, b)
            new_costs = cost_value(α, b)
            best_action = α.action
        end
    end
    return best_action, new_val, new_costs
end

function update_V_upper(tree::RecursivelyConstrainedPBVITree, b_idx::Int, d::AbstractVector)
    Vf = -Inf
    V = -Inf
    C = fill(Inf, length(tree.pomdp.constraints))
    b = tree.b[b_idx]
    for a ∈ actions(tree.pomdp)
        Qv = tree.Qa_upper[b_idx][a]
        Qc = tree.Qca_lower[b_idx][a]
        Qf = feasibility_value(tree.Qca_lower[b_idx][a], d)

        if Qf > Vf && Vf < 0.
            Vf = Qf
            C .= Qc
            V = Qv
        elseif Vf > 0. && Qv > V && Qf > 0.
            Vf = Qf
            C .= Qc
            V = Qv
        end
    end
    return V, C
end

function update_V_lower(tree::RecursivelyConstrainedPBVITree, b_idx::Int, d::AbstractVector)
    Vf = -Inf
    V = -Inf
    C = fill(Inf, length(tree.pomdp.constraints))
    best_a = first(actions(tree.pomdp))
    for a ∈ actions(tree.pomdp)
        Qv = tree.Qa_lower[b_idx][a]
        Qc = tree.Qca_upper[b_idx][a]
        Qf = feasibility_value(tree.Qca_upper[b_idx][a], d)

        if Qf > Vf && Vf < 0.
            Vf = Qf
            C .= Qc
            V = Qv
            best_a = a
        elseif Vf > 0. && Qv > V && Qf > 0.
            Vf = Qf
            C .= Qc
            V = Qv
            best_a = a
        end
    end
    return V, C, tree.b_children[b_idx][best_a]
end

function upper_feasibility(tree::RecursivelyConstrainedPBVITree, b::AbstractVector, d::AbstractVector)
    α_corner = tree.Cs_lower
    f_corner = feasibility_value(α_corner, b, d)
    C_lower = tree.C_lower
    f̂_min = Inf

    for b_idx ∈ tree.real
        (tree.b_pruned[b_idx] || tree.is_terminal[b_idx]) && continue
        b_int = tree.b[b_idx]
        c_int = feasibility_value(C_lower[b_idx], b_int, d)
        ϕ = min_ratio(b, b_int)
        f̂ = f_corner + ϕ * (c_int - feasibility_value(α_corner, b_int, d))
        f̂_min = min(f̂_min, f̂)
    end
    return f̂_min
end

root_diff(tree) = tree.V_upper[1] - tree.V_lower[1]
