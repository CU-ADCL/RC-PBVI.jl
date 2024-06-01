const RSPos = SVector{2, Int}

struct RSState{K}
    pos::RSPos 
    rocks::SVector{K, Bool}
end

@with_kw struct RockSamplePOMDP{K} <: POMDP{RSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 20.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    step_penalty::Float64 = 0.
    sensor_use_penalty::Float64 = 0.
    exit_reward::Float64 = 10.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
end

const N_BASIC_ACTIONS = 5
const BASIC_ACTIONS_DICT = Dict(:sample => 1,
                                :north => 2, 
                                :east => 3,
                                :south => 4,
                                :west => 5)

const ACTION_DIRS = (RSPos(0,0),
                    RSPos(0,1),
                    RSPos(1,0),
                    RSPos(0,-1),
                    RSPos(-1,0))

POMDPs.actions(pomdp::RockSamplePOMDP{K}) where K = 1:N_BASIC_ACTIONS+K
POMDPs.actionindex(pomdp::RockSamplePOMDP, a::Int) = a

function POMDPs.actions(pomdp::RockSamplePOMDP{K}, s::RSState) where K
    if in(s.pos, pomdp.rocks_positions) # slow? pomdp.rock_pos is a vec 
        return actions(pomdp)
    else
        # sample not available
        return 2:N_BASIC_ACTIONS+K
    end
end
