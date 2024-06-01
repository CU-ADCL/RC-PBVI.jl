Base.@kwdef struct RecursivelyConstrainedPBVISolver{LOW, UP} <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = Inf
    max_iter::Int       = 10
    init_lower::LOW     = BlindLowerBound(bel_res = 1e-1)
    init_upper::UP      = FastInformedBound(bel_res = 1e-1)
    verbose::Bool       = false
end

function POMDPTools.solve_info(solver::RecursivelyConstrainedPBVISolver, pomdp::POMDP)

    tree = RecursivelyConstrainedPBVITree(solver, pomdp)
    t0 = time()
    iter = 0
    while time()-t0 < solver.max_time && root_diff(tree) > solver.precision
        iter += 1
        if rand() < 0.5
            sample!(solver, tree)
        else
            sample_random!(solver, tree)
        end
        backup!(tree)
        prune!(tree)
    end
    return StatefulPBVIPolicy(pomdp, tree, solver), (;
        time = time() - t0,
        iter,
        tree
    )
end

POMDPs.solve(solver::RecursivelyConstrainedPBVISolver, pomdp::POMDP) = first(solve_info(solver, pomdp))
