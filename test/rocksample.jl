begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__,".."))
    using RecursivelyConstrainedPBVI
    Pkg.activate(@__DIR__)
    using POMDPs
    using ConstrainedPOMDPs
    using POMDPModels
    using ConstrainedPOMDPModels
    using POMDPTools
    include("Envs/Envs.jl")
    using .Envs
    using RockSample
    include("simulator.jl")
end

RS_SEED = 11878
_pomdp = RockSamplePOMDP((7,7), 8, Random.MersenneTwister(RS_SEED+1))
pomdp = RockSampleCPOMDP(_pomdp, [100.0])
pomdp = RockSampleCPOMDP()
sol = RecursivelyConstrainedPBVISolver(
    max_time = 10.0, 
    epsilon=1e-2,
    init_upper = RecursivelyConstrainedPBVI.FastInformedBound(max_time=5.0)
)
pol, info = solve_info(sol, pomdp)
info.tree.V_lower[1]
info.tree.V_upper[1]






using NativeSARSOP
sarsop_sol = SARSOPSolver(max_time=10.0)
sarsop_pol, sarsop_info = solve_info(sarsop_sol, _pomdp)
sarsop_info |> propertynames
sarsop_info.tree.V_lower[1]
sarsop_info.tree.V_upper[1]

value(sarsop_pol, initialstate(_pomdp))






sim = HistoryRecorder(max_steps=20)




value(sarsop_pol, initialstate(pomdp))
action(sarsop_pol, initialstate(pomdp))



init_upper = RecursivelyConstrainedPBVI.FastInformedBound(max_time=10.0)
tpomdp = TabularCPOMDP(pomdp)
upper_pol = solve(init_upper, tpomdp)
@profview solve(init_upper, tpomdp)
b0 = initialstate(tpomdp)
upper_pol[1] |> propertynames

pol = solve(sol, pomdp)

pol.tree.V_lower[1]
pol.tree.V_upper[1]


##

rs = RockSampleCPOMDP()
sol = RecursivelyConstrainedPBVISolver(max_time=1.)
pol = solve(sol,rs)

pol.tree.max_cost
first(pol.tree.b_guarantee_depth) == typemax(Int)
ĉ = [5.]
cpomdp = RockSampleCPOMDP(RockSamplePOMDP(5,7), ĉ)
# cpomdp = RockSampleCPOMDP(RockSamplePOMDP(15,15), ĉ)

tcpomdp = RecursivelyConstrainedPBVI.TabularCPOMDP(cpomdp)
tcpomdp.R

# cpomdp = RockSampleCPOMDP(ĉ, rs)
solver = RecursivelyConstrainedPBVI.RecursivelyConstrainedPBVISolver()
tree = RecursivelyConstrainedPBVI.RecursivelyConstrainedPBVITree(solver, cpomdp)
for _ ∈ 1:500
    RecursivelyConstrainedPBVI.sample!(solver,tree)
    RecursivelyConstrainedPBVI.backup!(tree)
end

policy = RecursivelyConstrainedPBVI.StatefulPBVIPolicy(cpomdp, tree)

policy = pol
cpomdp = rs
sim = ConstrainedDisplaySimulator(verbose = false, max_steps = 50)
v_hist = []
cost_hist = []

for _ ∈ 1:100
    RecursivelyConstrainedPBVI.reset!(policy)
    r_total, cost_tot = POMDPs.simulate(sim, cpomdp, policy)
    push!(v_hist, r_total)
    push!(cost_hist, cost_tot)
end

using Plots
using StatsPlots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

p1 = violin([""], v_hist; ylabel="reward", show_mean=true, bandwidth=0.05)
p2 = violin([""], cost_hist; ylabel="cost", c=:red, show_mean=true, bandwidth=0.05)
p2 = plot(cost_hist; ylabel="cost", yaxis=[0,1.0])
p = plot(p1,p2, dpi=300)

savefig(p, "RCPOMDP_rocksample78_violin.png")

##
using Distributed
addprocs(3)
# here v1 IS modified
v1 = zeros(100)
tmp1 = pmap(eachindex(v)) do i
    v1 .= 1
    2
end
v1
all(isone, v1) # true

#
v2 = zeros(100)
tmp2 = pmap(eachindex(v)) do i
    v2[i] = 1
    3
end
v2
all(isone, v2)
# oh no...
