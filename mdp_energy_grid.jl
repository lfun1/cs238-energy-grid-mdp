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

# Default Transition function T(s' | s, a)
function transitions_default(s::EnergyGridState, a::EnergyGridAction)
    likelihood = rand()
    if a.action == :addSupply && likelihood < 0.8
        add_vertex!(s.g)
    elseif a.action == :addStorage && likelihood < 0.7
        add_vertex!(s.g)
    elseif a.action == :addEdge && likelihood < 0.6
        add_edge!(s.g, a.edge[1], a.edge[2])
    end
end

# EnergyGridMDP constructor
function EnergyGridMDP(;
    g_init::Graph = Graph(),
    rewards::Function = rewards_default,
    transitions::Function = transitions_default,
    discount_factor::Float64=0.9)
    return EnergyGridMDP(g_init, rewards, transitions, discount_factor)
end




