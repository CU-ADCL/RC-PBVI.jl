function is_terminal_belief(pomdp::TabularCPOMDP, b)
    return dot(pomdp.isterminal, b) ≈ 1.0
end

function predictor(pomdp::TabularCPOMDP, b::SparseVector, a::Int)
    return predictor!(similar(b), pomdp, b, a)
end

function predictor!(cache, pomdp::TabularCPOMDP, b::SparseVector, a::Int)
    return mul!(cache, pomdp.T[a], b)
end

function corrector(pomdp::TabularCPOMDP, pred::AbstractVector, a, o::Int)
    return pred .* @view(pomdp.O[a][:,o])
end
