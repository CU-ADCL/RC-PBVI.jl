using Random

Base.@kwdef struct ConstrainedDisplaySimulator{RNG<:Random.AbstractRNG}
    max_steps::Int = 5
    max_fps::Float64 = 10.
    verbose::Bool = false
    rng::RNG = Random.default_rng()
end

sleep_until(t) = sleep(max(t-time(), 0.0))

function POMDPs.simulate(
        sim::ConstrainedDisplaySimulator,
        pomdp::RockSampleCPOMDP,
        policy,
        bu=DiscreteUpdater(pomdp)
    )
    s = rand(initialstate(pomdp))
    r_total = 0
    γ = discount(pomdp)
    d = only(pomdp.constraints)
    b = initialize_belief(bu, initialstate(pomdp))
    dt = inv(sim.max_fps)
    # display(render(pomdp, (sp=s, bp=b, a=first(actions(pomdp)))))
    cost_tot = only(zeros(constraint_size(pomdp)))
    tm = time()
    b_idx = 1
    o = rand(observations(pomdp))
    for i in 1:sim.max_steps
        isterminal(pomdp.m,s) && break

        a = action(policy, o)
        prevs = s
        c = only(costs(pomdp,s,a))
        s, o, r = @gen(:sp,:o,:r)(pomdp.m, s, a, sim.rng)
        sim.verbose && println("step: $i -- s: $prevs, a: $a, sp: $s")
        r_total += γ*r
        cost_tot += γ*c
        γ *= discount(pomdp)
        d = (d-c)/discount(pomdp)
        # display(render(pomdp, (sp=s, bp=b, a=a)))
        sleep_until(tm+dt)
        tm = time()
    end
    # sim.verbose && println("discounted reward: $r_total, discounted cost: $cost_tot")
    println("discounted reward: $r_total, discounted cost: $cost_tot")
    # @show r_total, cost_tot
    return r_total, cost_tot
end
