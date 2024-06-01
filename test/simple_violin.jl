begin
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    
    Pkg.activate(@__DIR__)
    # using CGCP
    using POMDPs
    using POMDPTools
    using POMDPModels
    using ConstrainedPOMDPs
    using ConstrainedPOMDPModels
    using ProgressMeter
    using StatsPlots
    using Plots
    include("Envs/Envs.jl")
    using .Envs
end

cpomdp = SimpleCPOMDP([6.05])

function POMDPs.solve(solver::CGCPSolver, pomdp::CPOMDP)
    t0 = time()
    (;max_time, max_iter, evaluator, verbose) = solver
    nc = constraint_size(pomdp)
    prob = CGCP.CGCPProblem(pomdp, ones(nc), false)

    τ = solver.τ 
    pomdp_solver = CGCP.HSVI4CGCP.SARSOPSolver(max_time=τ,max_steps=solver.max_steps,delta=0.75)
    # pomdp_solver = PBVISolver(max_time=0.0325,max_iter=typemax(Int))
    π0, v0, c0 = CGCP.initial_policy(solver, prob, pomdp_solver)
    Π = [π0]
    V = [v0]
    C = reshape(c0, nc, 1)

    lp = CGCP.master_lp(solver, prob, C,V)
    CGCP.optimize!(lp)
    λ = CGCP.dual(lp[:CONSTRAINT])::Vector{Float64}
    λ_hist = [λ]

    πt = CGCP.compute_policy(pomdp_solver,prob,λ)
    v_t, c_t = CGCP.evaluate_policy(evaluator, prob, πt)

    iter = 1
    verbose && println("""
        iteration $iter
        c = $c_t
        v = $v_t
        λ = $λ
    ----------------------------------------------------
    """)
    
    while time() - t0 < max_time && iter < max_iter
        push!(Π, πt)
        push!(V, v_t)
        C = hcat(C, c_t)
        iter += 1

        lp = CGCP.master_lp(solver,prob,C,V)
        CGCP.optimize!(lp)
        λ = CGCP.dual(lp[:CONSTRAINT])::Vector{Float64}
        push!(λ_hist, λ)

        ϕl = CGCP.JuMP.objective_value(lp) 
        ϕu = CGCP.dot(λ,constraints(pomdp))

        if λ == λ_hist[end-1]
            τ += solver.τ_inc
        end 

        δ = maximum(abs, λ .- λ_hist[end-1])
        
        pomdp_solver = CGCP.HSVI4CGCP.SARSOPSolver(max_time=τ,max_steps=solver.max_steps,delta=0.75, prunethresh=Inf)
        # pomdp_solver = PBVISolver(max_time=0.0325,max_iter=typemax(Int))
        πt = CGCP.compute_policy(pomdp_solver,prob,λ)
        v_t, c_t = CGCP.evaluate_policy(evaluator, prob, πt)
        ϕu += POMDPs.value(πt,initialstate(pomdp))
        ϕa = 10^(log10(max(abs(ϕl),abs(ϕu)))-solver.ρ)

        verbose && println("""
            iteration $iter
            c = $c_t
            v = $v_t
            λ = $λ
            δ = $δ
            ϕa = $ϕa
            Δϕ = $(ϕu-ϕl)
        ----------------------------------------------------
        """)
        ((ϕu-ϕl)<ϕa) && break
        # δ == 0 && break
    end
    return CGCP.CGCPSolution(Π, CGCP.JuMP.value.(lp[:x]), lp, C, V, λ_hist, 0, prob, evaluator)
end

sol = CGCPSolver()
pol = solve(sol, cpomdp)

value(pol, initialstate(cpomdp))

N = 100_000
sim_rewards = zeros(N)
sim_costs = zeros(N)

@showprogress for i ∈ 1:N
    p_idx = rand(SparseCat(eachindex(pol.policy_vector), pol.p_pi))
    policy = pol.policy_vector[p_idx]
    sim = RolloutSimulator(max_steps = 20)
    r,c = simulate(sim, cpomdp, policy, DiscreteUpdater(cpomdp), initialstate(cpomdp))
    sim_rewards[i] = r
    sim_costs[i] = only(c)
end

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

p1 = violin([""], sim_rewards; ylabel="reward", show_mean=true, bandwidth=1.0)
p2 = violin([""], sim_costs; ylabel="cost", c=:red, show_mean=true, bandwidth=0.05)
p = plot(p1,p2, dpi=300)

savefig(p, "CGCP_violin.png")

pol.p_pi

value(pol.policy_vector[4], initialstate(cpomdp))
value(pol.policy_vector[5], initialstate(cpomdp))

action(pol.policy_vector[5], initialstate(cpomdp))
action(pol.policy_vector[4], initialstate(cpomdp))

initialstate(cpomdp)
sp, o, r = @gen(:sp, :o, :r)(cpomdp, 1, 1)
sp, o, r = @gen(:sp, :o, :r)(cpomdp, 2, 1)

up = DiscreteUpdater(cpomdp)
bp = update(up, initialstate(cpomdp),1,false)
action(pol.policy_vector[4], bp)

bp = update(up, initialstate(cpomdp),1,true)
action(pol.policy_vector[4], bp)
