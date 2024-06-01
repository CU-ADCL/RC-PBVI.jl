struct ClosedLoop{SOL, P} <: Policy
    sol::SOL
    sparse_pomdp::TabularCPOMDP
    pomdp::P
end

ClosedLoop(sol, pomdp::POMDP) = ClosedLoop(sol, TabularCPOMDP(pomdp), pomdp)

POMDPTools.DiscreteBelief(p::CPOMDP, S::Base.OneTo{Int}, b) = DiscreteBelief(p, Array(S), b)

function POMDPs.action(p::ClosedLoop, b, d)
    copyto!(p.sparse_pomdp.initialstate, _initialize_belief(p.pomdp, b))
    copyto!(p.sparse_pomdp.constraints, d)
    pol = solve(p.sol, p.sparse_pomdp)
    a_idx = action(pol, initialstate(p.sparse_pomdp))
    return ordered_actions(p.pomdp)[a_idx]
end
