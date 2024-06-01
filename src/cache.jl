struct TreeCache
    pred::SparseVector{Float64,Int}
    v_alpha::Vector{Float64}
    c_alpha::Matrix{Float64}
    Γ::Array{Float64,3}
    Γc::Array{Float64, 4}
    Oᵀ::Vector{SparseMatrixCSC{Float64, Int64}}
end

function TreeCache(pomdp::TabularCPOMDP)
    Ns = n_states(pomdp)
    Na = n_actions(pomdp)
    No = n_observations(pomdp)
    Nc = ConstrainedPOMDPs.constraint_size(pomdp)

    pred = Vector{Float64}(undef, Ns)
    v_alpha = Vector{Float64}(undef, Ns)
    c_alpha = Matrix{Float64}(undef, Ns, Nc)
    Γ = Array{Float64,3}(undef, Ns, No, Na)
    Γc = Array{Float64, 4}(undef, Ns, No, Na, Nc)
    Oᵀ = map(sparse ∘ transpose, pomdp.O)
    return TreeCache(pred, v_alpha, c_alpha, Γ, Γc, Oᵀ)
end
