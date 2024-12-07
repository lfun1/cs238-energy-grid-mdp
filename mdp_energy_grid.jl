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
    budget::Int64   # Budget remaining to add vertices or edges
end

# EnergyGrid Actions
struct EnergyGridAction
    action::Symbol      # Actions include :addSupply, :addStorage, :addEdge, :noAction
    edge::Union{Tuple{Int, Int}, Nothing}   # Specify 2 vertices for :addEdge action
end

# EnergyGridMDP type
struct EnergyGridMDP <: MDP{EnergyGridState, EnergyGridAction}
    g_init::Graph   # Intial grid configuration, includes initial number of vertices
    action_cost::Dict{Symbol, Int64}   # Dictionary mapping action symbol to cost
    reward_reliability::Function    # Function to evaluate reliability of state's graph (Karger's Algorithm)
    reward_alpha::Float64   # Relative weighting of reliability vs budget in reward
    reward_beta::Float64   # Relative number of vertices vs budget in reward
    discount_factor::Float64 # discount factor
    # rewards::Function       # Reward function mapping states and actions to reward (params: number of storage/supply vertices)
    # transitions::Function   # Transition function that modifies state based on action with certain probabilities
end

# v1 Default Reward function R(s, a)
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
    next_state_prob = POMDPs.transition(mdp, s::EnergyGridState, a::EnergyGridAction)
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

# v2 Default Transition function T(s' | s, a)
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

"""
Function to run Karger's Algorithm for many trials to achieve higher probability of success.
    Only applies to unweighted undirected graphs with at most 1 edge between vertices.
    Uses Graphs.karger_min_cut(g) to for each run of Karger's Algorithm.
"""
function karger_multiple_min_cut(g::Graph; prob_fail::Float64=0.01)
    # Given delta = probability of failure
    # Number of trials T >= n^2 * ln(1/delta) >= (n choose 2) * ln(1/deta)
    num_trials = trunc(Int, nv(g) * nv(g) * log(1/prob_fail)) + 1
    min_cut_cost = nv(g) * (nv(g) - 1)   # Maximum cut value = maximum number of edges

    # Repeat Karger's algorithm num_trials times
    for i in 1:num_trials
        cut = karger_min_cut(g)
        cut_cost = karger_cut_cost(g, cut)
        min_cut_cost = min(min_cut_cost, cut_cost)
    end

    return min_cut_cost
end

# EnergyGridMDP constructor
function EnergyGridMDP(;
    g_init::Graph = SimpleGraph(),
    action_cost::Dict{Symbol, Int64} = Dict(
        :addSupply => 1, :addStorage => 1, :addEdge => 3, :noAction => 0
    ),
    reward_reliability::Function = karger_multiple_min_cut,
    reward_alpha::Float64 = 30.0,
    reward_beta::Float64 = 4.0,
    discount_factor::Float64=0.9)
    return EnergyGridMDP(g_init, action_cost, reward_reliability, reward_alpha, reward_beta, discount_factor)
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
function POMDPs.transition(mdp::EnergyGridMDP, s::EnergyGridState, a::EnergyGridAction) 
    g_copy = copy(s.g)
    budget_remain = s.budget - mdp.action_cost[a.action]
    action_probs = Dict(:addSupply => 0.8, :addStorage => 0.7, :addEdge => 0.6, :noAction => 1.0)
    if a.action == :noAction || budget_remain < 0
        # State remains same, no action taken
        return SparseCat([EnergyGridState(g_copy, budget_remain)], [1.0])
    elseif a.action in [:addSupply, :addStorage]
        add_vertex!(g_copy)
    elseif a.action == :addEdge
        add_edge!(g_copy, a.edge[1], a.edge[2])
    else
        error("Invalid action")
    end
    return SparseCat([EnergyGridState(g_copy, budget_remain), EnergyGridState(copy(s.g), budget_remain)], 
                    [action_probs[a.action], 1 - action_probs[a.action]])
end

# Reward function: based on reliability, number of vertices, action
# Added budget to for proper spending behavior
function POMDPs.reward(mdp::EnergyGridMDP, s::EnergyGridState, a::EnergyGridAction)
    r_reliability = mdp.reward_reliability(s.g)
    r_vertices = nv(s.g)

    # Initial reward function:
    return mdp.reward_alpha * r_reliability + r_vertices

    # Budget reward function:
    
    # println("Reward components: min cut = ", r_reliability, ", vertices = ", r_vertices)
    return mdp.reward_alpha * r_reliability + mdp.reward_beta * r_vertices + s.budget
end

# Discount
function POMDPs.discount(mdp::EnergyGridMDP)
    return mdp.discount_factor
end


"""
Evaluate MDP with MCTS
"""

# Single full run of MCTS online planning
function evaluate_mdp_2a(mdp::EnergyGridMDP, s0::EnergyGridState, num_steps::Int64, )
    solver = MCTSSolver(n_iterations=100, depth=10, exploration_constant=10.0)
    mcts_planner = POMDPs.solve(solver, mdp)

    # Metrics to track
    metrics::Dict{Symbol, Any} = Dict(
        :reliability => 0,
        :total_cost => 0,
        :total_steps => 0,
        :max_degree => 0,
        :total_reward => 0,
        :discounted_reward => 0,
        :final_graph => SimpleGraph()
    )

    s = s0

    for i in 1:num_steps
        println("Step ", i)

        a = action(mcts_planner, s)
        println("Action a: ", a)
        metrics[:total_cost] += mdp.action_cost[a.action]

        s, r = @gen(:sp, :r)(mdp, s, a, mcts_planner.rng)
        println("Received reward: ", r)
        metrics[:total_reward] += r
        metrics[:discounted_reward] += mdp.discount_factor ^ (i-1) * r

        println("New state sp: ", s)
        if metrics[:total_steps] == 0 && s.budget <= 0
            metrics[:total_steps] = i
        end

        println()
    end

    # Compute metrics
    metrics[:reliability] = mdp.reward_reliability(s.g)
    metrics[:max_degree] = maximum(degree(s.g))
    metrics[:final_graph] = copy(s.g)

    return metrics
end

# Evaluate MDP with different hyperparameters
function evaluate_mdp_2()
    alpha_r_list = [2.0, 5.0, 8.0]
    budget_list = [10, 20, 50]
    num_steps = 20

    experiments = Dict()
    
    for alpha_r in alpha_r_list
        for budget in budget_list
            mdp = EnergyGridMDP(reward_alpha=alpha_r)
            s0 = EnergyGridState(SimpleGraph(), budget)     # Budget
            println("New run")
            println("Alpha_r = ", alpha_r, ". Budget = ", budget)
            metrics = evaluate_mdp_2a(mdp, s0, num_steps)
            println("Metrics:")
            println(metrics)
            experiments[(alpha_r, budget)] = metrics
            println()
        end
    end

    println()
    println("Experiments: ")
    println(experiments)
end

evaluate_mdp_2()


""" Testing v2 """
function evaluate_mdp_1()
    mdp = EnergyGridMDP(reward_alpha=30.0, reward_beta=4.0)
    solver = MCTSSolver(n_iterations=200, depth=20, exploration_constant=10.0)
    mcts_planner = POMDPs.solve(solver, mdp)

    s0 = EnergyGridState(SimpleGraph(), 100)     # Budget
    num_steps = 100
    s = s0

    # Metrics to track
    metrics::Dict{Symbol, Any} = Dict(
        :reliability => 0,
        :total_cost => 0,
        :total_steps => 0,
        :max_degree => 0,
        :total_reward => 0,
        :discounted_reward => 0    
    )

    action_order = Vector{EnergyGridAction}()

    for i in 1:num_steps
        println("Step ", i)

        a = action(mcts_planner, s)
        println("Action a: ", a)
        push!(action_order, a)
        if metrics[:total_steps] == 0
            metrics[:total_cost] += mdp.action_cost[a.action]
        end

        global s, r = @gen(:sp, :r)(mdp, s, a, mcts_planner.rng)
        println("Received reward: ", r)
        metrics[:total_reward] += r
        metrics[:discounted_reward] += mdp.discount_factor ^ (i-1) * r

        println("New state sp: ", s)
        if metrics[:total_steps] == 0 && s.budget <= 0
            metrics[:total_steps] = i
        end

        println()
    end

    # Compute metrics
    metrics[:reliability] = mdp.reward_reliability(s.g)
    metrics[:max_degree] = maximum(degree(s.g))

    println("List of actions: ")
    println(action_order)
    # println([(i, a.action) for (i, a) in enumerate(action_order)])

    println("Metrics: ")
    println(metrics)

    println("Degrees: ", degree(s.g))
    
end

"""Initial testing"""
function evaluate_mdp_0()
    # print(actions(mdp, EnergyGridState(SimpleGraph(5))))

    solver = MCTSSolver(n_iterations=100, depth=10, exploration_constant=10.0)
    mcts_planner = POMDPs.solve(solver, mdp)

    s0 = EnergyGridState(SimpleGraph(5), 12)
    s = s0
    total_reward = 0
    num_steps = 10
    action_order = Vector{EnergyGridAction}()
    for i in 1:num_steps
        println("Step ", i)
        a = action(mcts_planner, s)
        println("Action a: ", a)
        push!(action_order, a)
        global s, r = @gen(:sp, :r)(mdp, s, a, mcts_planner.rng)
        global total_reward += r
        println("New state sp: ", s)
        println()
    end

    println("List of actions: ", [(i, a.action) for (i, a) in enumerate(action_order)])
    println("Total reward: ", total_reward)

# Debugging outputs:
# print(mcts_planner.rng)
# print(@gen(:sp, :r)(mdp, s0, EnergyGridAction(:addSupply, nothing), mcts_planner.rng))
end