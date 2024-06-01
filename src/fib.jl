#=
If we max(vr) and min(vc) then they are definitely sound bounds
=#
Base.@kwdef mutable struct FastInformedBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    init_value::Float64         = 0.
    α_tmp::Vector{Float64}      = Float64[]
    c_tmp::Matrix{Float64}      = Matrix{Float64}(undef,0,0)
    residuals::Vector{Float64}  = Float64[]
end

function bel_res(α1, α2)
    max_res = 0.
    @inbounds for i ∈ eachindex(α1, α2)
        res = abs(α1[i] - α2[i])
        res > max_res && (max_res = res)
    end
    return max_res
end

function update!(𝒫::TabularCPOMDP, M::FastInformedBound, Γ)
    (;R,T,O,C) = 𝒫
    d = 𝒫.constraints
    Nc = n_constraints(𝒫)
    γ = discount(𝒫)
    residuals = M.residuals


    # FIXME: GC
    tmp_c = Vector{Float64}(undef, Nc)
    Cmin = Vector{Float64}(undef, Nc)
    Cb′ = Vector{Float64}(undef, Nc)

    for a ∈ actions(𝒫)
        α_a_r = M.α_tmp
        α_a_c = M.c_tmp
        T_a = T[a]
        O_a = O[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)

        for s ∈ states(𝒫)
            rsa = R[s,a]
            csa = @view C[s,a,:]

            if isterminal(𝒫,s)
                α_a_r[s] = 0.
                α_a_c[s,:] .= 0.
            else
                tmp_r = 0.0
                tmp_c .= 0.0
                for o ∈ observations(𝒫)
                    O_ao = @view O_a[:,o]
                    Vmax = -Inf
                    Fmax = -Inf
                    Cmin .= Inf
                    for α′ ∈ Γ
                        Vb′ = 0.0
                        Fb′ = 0.0
                        Cb′ .= 0.0
                        for idx ∈ nzrange(T_a, s)
                            sp = rv[idx]
                            Tprob = nz[idx]
                            p = O_ao[sp]*Tprob # FIXME: slow sparse indexing
                            Vb′ += p*α′.v_alpha[sp]
                            @views @. Cb′ .+= p*α′.c_alpha[sp,:]
                        end
                        Fb′ = feasibility_value(Cb′, d)
                        Vb′ > Vmax && (Vmax = Vb′)
                        if Fb′ > Fmax # can't do pointwise min - use feasibility criterion instead
                            Cmin .= Cb′
                            Fmax = Fb′
                        end
                    end
                    tmp_r += Vmax
                    tmp_c .+= Cmin
                end
                α_a_r[s] = rsa + γ*tmp_r
                @. α_a_c[s,:] = csa + γ*tmp_c
            end
        end
        res1 = bel_res(Γ[a].v_alpha, α_a_r)
        res2 = bel_res(Γ[a].c_alpha, α_a_c)
        residuals[a] = max(res1, res2)
        copyto!(Γ[a].v_alpha, α_a_r)
        copyto!(Γ[a].c_alpha, α_a_c)
    end
end

POMDPs.solve(sol::FastInformedBound, pomdp::POMDP) = solve(sol,TabularCPOMDP(pomdp))

function POMDPs.solve(sol::FastInformedBound, pomdp::TabularCPOMDP)
    t0 = time()
    S = states(pomdp); Ns = n_states(pomdp)
    A = actions(pomdp); Na = n_actions(pomdp)
    O = observations(pomdp); No = n_observations(pomdp)
    Nc = n_constraints(pomdp)
    γ = discount(pomdp)

    init_value = sol.init_value
    rmax = maximum(pomdp.R) / (1 - γ)
    Γ = [AlphaVec(fill(rmax, Ns), zeros(Ns, Nc), a) for a ∈ A]
    resize!(sol.α_tmp, Ns)
    sol.c_tmp = Matrix{Float64}(undef, Ns, Nc)
    residuals = resize!(sol.residuals, Na)

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return Γ
end

function max_upper_val(Γ::Vector{<:AlphaVec})
    max_alpha = first(Γ).v_alpha
    for α ∈ @view(Γ[2:end])
        @. max_alpha = max(max_alpha, α.v_alpha)
    end
    return max_alpha
end

function min_lower_constraint(Γ::Vector{<:AlphaVec})
    min_alpha = first(Γ).c_alpha
    for α ∈ @view(Γ[2:end])
        @. min_alpha = min(min_alpha, α.c_alpha)
    end
    return min_alpha
end
