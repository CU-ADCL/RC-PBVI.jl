using RecursivelyConstrainedPBVI
opt = RecursivelyConstrainedPBVI.MaxMinOptimizer()
ns = 10
nc = 1
nα = 4
Γ = [RecursivelyConstrainedPBVI.AlphaVec(rand(ns),rand(ns, nc),1) for _ in 1:nα]
push!(Γ, RecursivelyConstrainedPBVI.AlphaVec(rand(ns),zeros(ns, nc), 1))
RecursivelyConstrainedPBVI.alphaset_matrix(Γ)
mmc = RecursivelyConstrainedPBVI.max_min_cost(Γ)
