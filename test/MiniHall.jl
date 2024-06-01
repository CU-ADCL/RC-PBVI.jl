using POMDPs
using POMDPModels
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using Test
using LinearAlgebra

@testset "tiger" begin
    ĉ = [1.0]

    c_tiger = constrain(TigerPOMDP(), ĉ) do s,a
        iszero(a) ? [1.0] : [0.0]
    end


end

# ConstrainedPOMDPs.costs(m::typeof(c_tiger), s, a) = iszero(a) ? [1.0] : [0.0]

solver = RecursivelyConstrainedPBVISolver()
tree = RecursivelyConstrainedPBVITree(solver, c_tiger)

RecursivelyConstrainedPBVI.sample!(solver,tree)