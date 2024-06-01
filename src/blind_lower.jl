"""
Blind Lower bound initialization
"""
Base.@kwdef mutable struct BlindLowerBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    α_v_tmp::Vector{Float64}    = Float64[]
    α_c_tmp::Matrix{Float64}    = Matrix{Float64}(undef,0,0)
    residuals::Vector{Float64}  = Float64[]
end

function update!(pomdp::TabularCPOMDP, M::BlindLowerBound, Γ::Vector{Tuple{Vector{Float64},Matrix{Float64}}})
    residuals = M.residuals
    (;T,R,O,C) = pomdp
    γ = discount(pomdp)

    α_c_tmp = zeros(n_constraints(pomdp))

    for a ∈ actions(pomdp)
        α_v_a, A_c_a = Γ[a]
        α_a = M.α_v_tmp
        α_c = M.α_c_tmp
        fill!(α_c, 0.0)
        T_a = T[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)
        for s ∈ states(pomdp)
            Vb′ = 0.0
            fill!(α_c_tmp, 0.0)
            for idx ∈ nzrange(T_a, s)
                sp = rv[idx]
                p = nz[idx]
                Vb′ += p*Γ[a][1][sp]
                @. @views α_c_tmp += p * A_c_a[sp, :]
            end
            α_a[s] = R[s,a] + γ*Vb′
            @. α_c[s,:] = C[s,a] + γ*α_c_tmp
        end
        res = max(bel_res(Γ[a][1], α_a), bel_res(Γ[a][2], α_c))
        residuals[a] = res
        copyto!(α_v_a, α_a)
        copyto!(A_c_a, α_c)
    end
    return Γ
end

function worst_state_alphas(pomdp::TabularCPOMDP)
    (;R,T,C) = pomdp
    γ = discount(pomdp)
    Ns = n_states(pomdp)
    Na = n_actions(pomdp)
    No = n_observations(pomdp)
    Nc = n_constraints(pomdp)

    c_min = fill(-Inf,Nc)

    # (α_v, α_c) ∈ Γ
    Γ = [(zeros(Ns),zeros(Ns,Nc)) for _ in 1:Na]
    for a ∈ actions(pomdp)
        nz = nonzeros(T[a])
        rv = rowvals(T[a])
        for s ∈ states(pomdp)
            r_min = Inf
            for idx ∈ nzrange(T[a], s)
                sp = rv[idx]
                p = nz[idx]
                r′ = p*R[sp, a]
                r_min = min(r′, r_min)
                for c_idx ∈ eachindex(c_min)
                    c′ = p*C[s,a,c_idx]
                    c_min[c_idx] = max(c_min[c_idx], c′)
                end
            end
            Γ[a][1][s] = R[s, a] + γ / (1 - γ) * r_min
            @. Γ[a][2][s,:] = @view(C[s,a,:]) + γ / (1 - γ) * c_min
            fill!(c_min, -Inf)
        end
    end
    return Γ
end

function POMDPs.solve(sol::BlindLowerBound, pomdp::TabularCPOMDP)
    t0 = time()
    Ns = n_states(pomdp)
    Na = n_actions(pomdp)
    No = n_observations(pomdp)
    Nc = n_constraints(pomdp)

    Γ = worst_state_alphas(pomdp)
    resize!(sol.α_v_tmp, Ns)
    sol.α_c_tmp = Matrix{Float64}(undef,Ns,Nc) # resize!(sol.α_c_tmp, Ns, Nc)
    residuals = resize!(sol.residuals, Na)

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return [AlphaVec(Γ[i][1], Γ[i][2], i) for i ∈ actions(pomdp)]
end
