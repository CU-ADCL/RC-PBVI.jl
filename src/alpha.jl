struct AlphaVec
    v_alpha::Vector{Float64} # [s_idx]
    c_alpha::Matrix{Float64} # [s_idx, c_idx]
    action::Int
end

@inline Base.length(v::AlphaVec) = length(v.v_alpha)

@inline Base.size(v::AlphaVec) = size(v.v_alpha)

@inline Base.getindex(v::AlphaVec, i) = v.v_alpha[i]

##

# NOTE: trying to MAXIMIZE feasibility
""" min(d - v) """
@inline function _feasibility_value(v::AbstractVector, d::AbstractVector)
    v_f = Inf
    for i ∈ eachindex(d,v)
        _vf = d[i] - v[i]
        _vf < v_f && (v_f = _vf)
    end
    return v_f
end

feasibility_value(v, d) = _feasibility_value(v,d)

function feasibility_value(A::Matrix, b::AbstractVector, d)
    v_f = Inf
    for j ∈ eachindex(d)
        c = 0.0
        for i ∈ eachindex(b)
            c += A[i,j]*b[i]
        end
        vf_j = d[j] - c
        vf_j < v_f && (v_f = vf_j)
    end
    return v_f
end

function feasibility_value(A::Matrix, b::SparseVector, d)
    v_f = Inf
    for j ∈ eachindex(d)
        c = 0.0
        nz_ind = b.nzind
        nz_val = b.nzval
        for _i ∈ eachindex(nz_ind)
            i = nz_ind[_i]
            c += A[i,j]*nz_val[_i]
        end
        vf_j = d[j] - c
        vf_j < v_f && (v_f = vf_j)
    end
    return v_f
end

@inline feasibility_value(v::AbstractVector,_,d::AbstractVector) = _feasibility_value(v,d)
@inline feasibility_value(α::AlphaVec, b, d) = feasibility_value(α.c_alpha, b, d)

@inline reward_value(α::Vector, b) = dot(α, b)
@inline reward_value(α::AlphaVec, b) = reward_value(α.v_alpha, b)

@inline cost_value(α::Matrix, b) = transpose(α)*b
@inline cost_value(α::AlphaVec, b) = cost_value(α.c_alpha, b)

@inline is_feasible(α::AlphaVec, b, d) = feasibility_value(α.c_alpha, b, d) > 0
