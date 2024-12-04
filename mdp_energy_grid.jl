"""
Formulate Markov Decision Process (MDP) for Renewable Energy Grid Application

Author: Lisa Fung
Last Updated: 12/3/2024
"""

using Graphs
using GraphPlot         # for plotting graphs
using Compose, Cairo, Fontconfig    # for saving graphs

using POMDPs, POMDPTools, MCTS
# using DiscreteValueIteration

# using Random

# EnergyGrid State
struct EnergyGridState
    g::Graph    # graph of energy grid configuration
end

# EnergyGrid Actions
struct EnergyGridAction
    action::Symbol      # Actions include :addSupply, :addStorage, :addEdge, :noAction
    edge::Union{Tuple{Int, Int}, Nothing}   # Specify 2 vertices for :addEdge action
end

# EnergyGridMDP type
struct EnergyGridMDP <: MDP{EnergyGridState, EnergyGridAction}
    g_init::Graph   # Intial grid configuration
    rewards::Function       # Reward function mapping states and actions to reward (params: number of storage/supply vertices)
    transitions::Function   # Transition function that modifies state based on action with certain probabilities
    discount_factor::Float64 # disocunt factor
end

# Default Reward function R(s, a)
function rewards_default(s::EnergyGridState, a::EnergyGridAction)
    # State
    r_s = nv(s.g)   # More supply/storage vertices, the better
    r_a = 0

    # Action
    if a.action == :addSupply
        r_a = -1
    elseif a.action == :addStorage
        r_a = -3
    elseif a.action == :addEdge
        r_a = -1
    end

    return r_s + r_a
end

# v1 Default Transition function T(s' | s, a)
# Directly updates state s to s', and returns s
function transition_state(s::EnergyGridState, a::EnergyGridAction)
    likelihood = rand()
    if a.action == :addSupply && likelihood < 0.8
        add_vertex!(s.g)
    elseif a.action == :addStorage && likelihood < 0.7
        add_vertex!(s.g)
    elseif a.action == :addEdge && likelihood < 0.6
        add_edge!(s.g, a.edge[1], a.edge[2])
    end
    return s
end

# Default Transition function T(s' | s, a)
# Returns a SparseCat (categorical distribution) instead of s'

function transitions_default(s::EnergyGridState, a::EnergyGridAction)
    g_copy = copy(s.g)
    action_probs = Dict(:addSupply => 0.8, :addStorage => 0.7, :addEdge => 0.6, :noAction => 1.0)
    if a.action in [:addSupply, :addStorage]
        add_vertex!(g_copy)
    elseif a.action == :addEdge
        add_edge!(g_copy, a.edge[1], a.edge[2])
    elseif a.action == :noAction
    else
        error("Invalid action")
    end
    return SparseCat([EnergyGridState(g_copy), EnergyGridState(copy(s.g))], [action_probs[a.action], 1 - action_probs[a.action]])
end

# EnergyGridMDP constructor
function EnergyGridMDP(;
    g_init::Graph = Graph(),
    rewards::Function = rewards_default,
    transitions::Function = transitions_default,
    discount_factor::Float64=0.9)
    return EnergyGridMDP(g_init, rewards, transitions, discount_factor)
end

# Skip state space (too large, using online planning techniques, e.g. MCTS)

# Actions from specific state s
POMDPs.actions(mdp::EnergyGridMDP, s::EnergyGridState) = 
    vcat([EnergyGridAction(a, nothing) for a in [:addSupply, :addStorage, :noAction]], 
         [EnergyGridAction(:addEdge, (i, j)) for i in 1:nv(s.g) for j in i+1:nv(s.g)]...)
# Note: allow adding edges that already exist in s.g::Graph for now, will not change graph

# Return action index
function POMDPs.actionindex(mdp::EnergyGridMDP, a::EnergyGridAction)
    @assert in(a, actions(mdp)) "Invalid action"
    return findfirst(x -> x == a, actions(mdp))
end

# Transition function
# Returns the transition distribution from the current state-action pair.
POMDPs.transition(mdp::EnergyGridMDP, s::EnergyGridState, a::EnergyGridAction) = mdp.transitions(s, a)

# Reward function (simple)
POMDPs.reward(mdp::EnergyGridMDP, s::EnergyGridState, a::EnergyGridAction) = mdp.rewards(s, a)

# Discount
function POMDPs.discount(mdp::EnergyGridMDP)
    return mdp.discount_factor
end


"""
Test POMDP code
"""
mdp = EnergyGridMDP()
# print(actions(mdp, EnergyGridState(SimpleGraph(5))))

solver = MCTSSolver(n_iterations=100, depth=10, exploration_constant=10.0)
mcts_planner = POMDPs.solve(solver, mdp)

s0 = EnergyGridState(SimpleGraph(5))
s = s0
num_steps = 20
action_order = Vector{EnergyGridAction}()
for i in 1:num_steps
    println("Step ", i)
    a = action(mcts_planner, s)
    println("Action a: ", a)
    push!(action_order, a)
    global s = transition_state(s, a)
    println("New state sp: ", s)
    println()
end

println("List of actions: ", [(i, a.action) for (i, a) in enumerate(action_order)])

# Debugging outputs:
# print(mcts_planner.rng)
# print(@gen(:sp, :r)(mdp, s0, EnergyGridAction(:addSupply, nothing), mcts_planner.rng))