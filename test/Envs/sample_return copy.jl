struct SandRoverPOMDP <: POMDP{GWPos,Symbol,GWPos}
    mdp::SimpleGridWorld
    obs_prob::Float64
    costs::Dict{GWPos, SVector{1,Float64}}
end

function SandRoverPOMDP(;obs_prob=0.9, costs=Dict{GWPos,SVector{1,Float64}}(), kwargs...)
    SandRoverPOMDP(SimpleGridWorld(;kwargs...),obs_prob, costs)
end

# POMDPs.transition(m::SandRoverPOMDP,s,a) = transition(m.mdp,s,a)
# POMDPs.reward(m::SandRoverPOMDP,s,a) = reward(m.mdp,s,a)
POMDPs.discount(m::SandRoverPOMDP) = discount(m.mdp)
POMDPs.actionindex(m::SandRoverPOMDP, a) = actionindex(m.mdp, a)
POMDPs.stateindex(m::SandRoverPOMDP, s) = stateindex(m.mdp, s)
POMDPs.actions(m::SandRoverPOMDP) = actions(m.mdp)
# POMDPs.actions(m::SandRoverPOMDP) = (:operator, :sand, :nosand)
POMDPs.states(m::SandRoverPOMDP) = states(m.mdp)
POMDPs.initialstate(m::SandRoverPOMDP) = Uniform((GWPos(2,2), GWPos(2,1))) #Deterministic(GWPos(1,2))
POMDPTools.ordered_states(m::SandRoverPOMDP) = ordered_states(m.mdp)
# POMDPs.actions(mdp::SimpleGridWorld) = (:operator, :sand, :nosand)
# POMDPs.actionindex(m::SandRoverPOMDP, a::Symbol) = aind[a]
POMDPs.actions(pomdp::SandRoverPOMDP, s) = actions(pomdp)
POMDPTools.ordered_actions(m::SandRoverPOMDP) = ordered_actions(m.mdp)
POMDPs.observations(w::SandRoverPOMDP) = states(w.mdp)
POMDPTools.ordered_observations(m::SandRoverPOMDP) = observations(m)
POMDPs.isterminal(m::SandRoverPOMDP,s) = POMDPs.isterminal(m.mdp,s)

function POMDPs.transition(m::SandRoverPOMDP, s, a)
    if s in m.mdp.terminate_from || isterminal(m, s) || s[1] == 5 || s[1] == 6
        return Deterministic(GWPos(-1,-1))
    end

    if s[1] == 1
        if a == :right || a == :up
            return Deterministic(GWPos(2, s[2]))
        else
            return Deterministic(GWPos(s[1], s[2]))
        end
    end

    competency = s[2]

    if competency == 1 #competent
        if a == :up #operator action 
            return Deterministic(GWPos(-1,-1)) #Deterministic(GWPos(4, competency))#
        elseif a == :right && s[1] == 2 # go to sand
            return Deterministic(GWPos(3, competency))
        elseif a == :right && s[1] == 3 # go to sand
            return Deterministic(GWPos(5, competency))
        elseif a == :right && s[1] == 4 # go to sand
            return Deterministic(GWPos(5, competency))
        elseif a == :down #go around sand
            return Deterministic(GWPos(6, competency))
        end
    else
        if a == :up
            return Deterministic(GWPos(-1,-1)) #Deterministic(GWPos(4, competency))
        elseif a == :right && s[1] == 2 # go to sand
            return Deterministic(GWPos(3, competency))
        elseif a == :right && s[1] == 3 # go through sand
            prob = m.mdp.tprob
            probs = Vector(zeros(2))
            probs[1] = prob
            probs[2] = 1 - prob
            destinations = MVector{2, GWPos}(undef)
            destinations[1] = GWPos(5, competency)
            destinations[2] = GWPos(4, competency)
            return SparseCat(destinations, probs./sum(probs))
        elseif a == :right && s[1] == 4 # go through sand
            prob = m.mdp.tprob
            probs = Vector(zeros(2))
            probs[1] = prob
            probs[2] = 1 - prob
            destinations = MVector{2, GWPos}(undef)
            destinations[1] = GWPos(5, competency)
            destinations[2] = GWPos(4, competency)
            return SparseCat(destinations, probs./sum(probs))
        elseif a == :down && s[1] == 2#go around sand
            return Deterministic(GWPos(6, competency))
        elseif a == :down && s[1] == 3
            prob = m.mdp.tprob
            probs = Vector(zeros(2))
            probs[1] = prob
            probs[2] = 1 - prob
            destinations = MVector{2, GWPos}(undef)
            destinations[1] = GWPos(6, competency)
            destinations[2] = GWPos(4, competency)
            return SparseCat(destinations, probs./sum(probs))
        elseif a == :down && s[1] == 4
            prob = m.mdp.tprob
            probs = Vector(zeros(2))
            probs[1] = prob
            probs[2] = 1 - prob
            destinations = MVector{2, GWPos}(undef)
            destinations[1] = GWPos(6, competency)
            destinations[2] = GWPos(4, competency)
            return SparseCat(destinations, probs./sum(probs))
        elseif a == :down
            return Deterministic(GWPos(6, competency))
        end
    end
    if a == :left
        return Deterministic(GWPos(4, competency))
    end

    @show a, s
end

function POMDPs.observation(m::SandRoverPOMDP,a,sp)
    isterminal(m, sp) && return Deterministic(sp)

    if a == :right && sp[1] == 3
        if sp[2] == 1
            return SparseCat((GWPos(sp[1], 1),GWPos(sp[1], 2),GWPos(sp[1], 3)), (m.obs_prob, 1-m.obs_prob, 0))
        else
            return SparseCat((GWPos(sp[1], 1),GWPos(sp[1], 2),GWPos(sp[1], 3)), (1-m.obs_prob, m.obs_prob, 0))
        end
    end

    return SparseCat((GWPos(sp[1], 1),GWPos(sp[1], 2),GWPos(sp[1], 3)), (0, 0, 1))
end

POMDPs.obsindex(m::SandRoverPOMDP, o) = stateindex(m,o)

const ConstrainedSandRover = typeof(ConstrainedPOMDPs.Constrain(SandRoverPOMDP(),[0.0]))

function ConstrainedSandRoverPOMDP(d::AbstractVector=[2.0]; kwargs...)
    return ConstrainedPOMDPs.Constrain(SandRoverPOMDP(;kwargs...), d)
end

function ConstrainedPOMDPs.cost(constrained::ConstrainedSandRover, s, a)
    pomdp = constrained.m
    mdp = pomdp.mdp

    return if s[1] == 4
        SA[1.0]
    else
        get(pomdp.costs, s, SA[0.0])
    end
end

function POMDPs.reward(m::SandRoverPOMDP,s,a)
    r = 0
    if s[1] == 5
        r = 100
    elseif s[1] == 6
        r = 10
    end
    return r
end

function POMDPTools.ModelTools.render(c_pomdp::ConstrainedSandRover, step)
    pomdp = c_pomdp.m
    mdp = pomdp.mdp
    nx, ny = mdp.size
    ny = ny-1
    cells = Context[]

    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), mdp.size)
        pos_rew = reward(c_pomdp, GWPos(x,y), step[:a]) > 0.
        color = if cost(c_pomdp, GWPos(x,y), step[:a]) == SA[1.0]
            pos_rew ? "blue" : "red"
        else
            pos_rew ? "blue" : "black"
        end
        target = compose(context(), rectangle(), fill(color), stroke("gray"))
        compose!(cell, target)
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), fill("black"), stroke("gray"))

    robot = if haskey(step, :sp)
        robot_ctx = cell_ctx(step[:sp], mdp.size)
        compose(robot_ctx, circle(0.5, 0.5, 0.5), fill("green"))
    else
        nothing
    end
    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), robot, grid, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    ny = ny - 1
    x, y = xy
    return context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
end

function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end