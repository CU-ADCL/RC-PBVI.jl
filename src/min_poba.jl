function basis(n,i)
    e_i = zeros(n)
    e_i[i] = 1
    return e_i
end

function min_poba(pomdp::TabularCPOMDP)
    _min_poba = 0.0
    _min_a = 0
    _min_o = 0
    _min_i = 0

    for a ∈ actions(pomdp)
        T = pomdp.T[a]
        for o ∈ observations(pomdp)
            ps = pomdp.O[a][:,o]' * T
            i = argmin(ps)
            poba = ps[i]
            if poba < _min_poba
                _min_poba = poba
                _min_a = a
                _min_o = o
                _min_i = i
            end
        end
    end
    _min_b = basis(n_states(pomdp), _min_i)
    return _min_poba, (_min_b,_min_a,_min_o)
end

function min_poba(pomdp::TabularCPOMDP, b)
    _min_poba = 0.0
    _min_a = 0
    _min_o = 0

    for a ∈ actions(pomdp)
        T = pomdp.T[a]
        for o ∈ observations(pomdp)
            poba = pomdp.O[a][:,o]' * T * b
            if poba < _min_poba
                _min_poba = poba
                _min_a = a
                _min_o = o
            end
        end
    end
    return _min_poba, (_min_a,_min_o)
end
