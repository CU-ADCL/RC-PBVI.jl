begin
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    using RecursivelyConstrainedPBVI
    Pkg.activate(@__DIR__)
    using POMDPs
    using POMDPTools
    using POMDPModels
    using ConstrainedPOMDPModels
    using ConstrainedPOMDPs
end

solver = RecursivelyConstrainedPBVISolver()

cpomdp = constrain(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.75), [3.0]) do s,a
    iszero(a) ? [1.0] : [0.0]
end

tree = RecursivelyConstrainedPBVITree(solver, cpomdp)

pol = RecursivelyConstrainedPBVI.StatefulPBVIPolicy(cpomdp, tree)
action(pol) # first action at root requires no observation
action(pol,false) # all following actions require obs to update internal state
