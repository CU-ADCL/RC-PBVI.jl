using RecursivelyConstrainedPBVI
import RecursivelyConstrainedPBVI as RCPBVI
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using Test

@testset "smoke" begin
    cpomdp = RockSampleCPOMDP()
    sol = RecursivelyConstrainedPBVISolver(max_time=1.0)
    pol = solve(sol, cpomdp)
    @test iszero(pol.tree.max_cost)
    @test first(pol.tree.b_guarantee_depth) == typemax(Int)
end
