struct MaxMinOptimizer{SOL,KWS}
    solver::SOL
    kwargs::KWS
    MaxMinOptimizer(sol=HiGHS.Optimizer;kwargs...) = new{typeof(sol), typeof(kwargs)}(sol, kwargs)
end


# α_c ∈ R^{ns × nc}
alphaset_matrix(Γ::AbstractVector{AlphaVec}) = alphaset_matrix(map(α->α.c_alpha, Γ))
function alphaset_matrix(Γ::AbstractVector{<:AbstractMatrix})
    @assert isone(size(first(Γ), 2)) # doesn't work with multiple costs
    return mapreduce(transpose, vcat, Γ)
end

function optimize(opt::MaxMinOptimizer, Γ)
    ns = length(first(Γ))
    Γ = alphaset_matrix(Γ)
    model = Model(
        optimizer_with_attributes(opt.solver, opt.kwargs...),
    )
    @variable(model, t)
    @objective(model, Max, t)
    @variable(model, b[1:ns])
    @constraint(model, Γ*b .≥ t)
    @constraint(model, b .≥ 0.)
    @constraint(model, sum(b) == 1.0)

    optimize!(model)

    # optimizer value sometimes returns -0.0 which messes with admissibility calculations
    return max(JuMP.value(t), 0.0)
end

function max_min_cost(Γ::AbstractVector; kwargs...)
    return optimize(MaxMinOptimizer(HiGHS.Optimizer; kwargs...), Γ) 
end
