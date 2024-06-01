function prune!(tree, explored=tree.sampled)
    for b_idx ∈ explored
        b_children = tree.b_children[b_idx]
        (tree.b_pruned[b_idx] || isempty(b_children)) && continue
        Qca_upper = tree.Qca_upper[b_idx]
        Qca_lower = tree.Qca_lower[b_idx]
        Qa_upper = tree.Qa_upper[b_idx]
        Qa_lower = tree.Qa_lower[b_idx]
        idxs = eachindex(Qca_upper,Qca_lower,Qa_upper,Qa_lower)
        pruned = falses(length(idxs))
        d = tree.admissible_cost[b_idx]

        for a ∈ idxs[1 : end-1]
            pruned[a] && continue
            ba_idx = b_children[a]
            kba = tree.ba_guarantee_depth[ba_idx]
            Qrba_upper = Qa_upper[a]
            Qrba_lower = Qa_lower[a]
            Qcba_upper = Qca_upper[a]
            for a′ ∈ idxs[a+1 : end]
                pruned[a′] && continue
                kbap = tree.ba_guarantee_depth[ba_idx]
                Qrbap_upper = Qa_upper[a′]
                Qrbap_lower = Qa_lower[a′]
                Qcbap_upper = Qca_upper[a′]
                if kba < kbap && Qrba_upper < Qrbap_lower && Qcba_upper < d
                    pruned[a] = true
                elseif kbap < kba && Qrbap_upper < Qrba_lower && Qcbap_upper < d
                    pruned[a′] = true
                end
            end
        end

        for a ∈ eachindex(pruned)
            pruned[a] && pruneSubTreeBa!(tree, ba_children[a])
        end
    end
end

function pruneSubTreeBa!(tree, ba_idx::Int)
    for b_idx in tree.ba_children[ba_idx]
        pruneSubTreeB!(tree, b_idx)
    end
    tree.ba_pruned[ba_idx] = true
end

function pruneSubTreeB!(tree, b_idx::Int)
    for ba_idx in tree.b_children[b_idx]
        pruneSubTreeBa!(tree, ba_idx)
    end
    tree.b_pruned[b_idx] = true
end
