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
end

cpomdp = SimpleCPOMDP([6.0])
@show discount(cpomdp)
solver = RecursivelyConstrainedPBVI.RecursivelyConstrainedPBVISolver()
tree = RecursivelyConstrainedPBVI.RecursivelyConstrainedPBVITree(solver, cpomdp)
for _ ∈ 1:500
    RecursivelyConstrainedPBVI.sample!(solver,tree)
    RecursivelyConstrainedPBVI.backup!(tree)
end

@show tree.V_lower[1], tree.V_upper[1]
@show tree.Qa_lower[1]
@show tree.Qca_upper[1]

policy = RecursivelyConstrainedPBVI.StatefulPBVIPolicy(cpomdp, tree)

sim = ConstrainedDisplaySimulator(verbose = false, max_steps = 20)
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

p1 = violin([""], v_hist; ylabel="reward", show_mean=true, bandwidth=1.0)
p2 = violin([""], cost_hist; ylabel="cost", c=:red, show_mean=true, bandwidth=0.05)
p = plot(p1,p2, dpi=300)

savefig(p, "RCPOMDP_simple_violin.png")