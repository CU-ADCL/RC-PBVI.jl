struct MinCostPOMDP{P<:CPOMDP,S,A,O} <: POMDP{S,A,O}
    cpomdp::P
    function MinCostPOMDP(p::CPOMDP)
        new{typeof(p), statetype(p), actiontype(p), obstype(p)}(p)
    end
end

@POMDP_forward MinCostPOMDP.pomdp

POMDPs.reward(p::MinCostPOMDP{P,S,A}, s::S, a::A) where {S,A} = -only(ConstrainedPOMDPs.cost(p.cpomdp, s, a))
POMDPs.reward(p::MinCostPOMDP, args...) = -only(ConstrainedPOMDPs.cost(p.cpomdp, args...))
