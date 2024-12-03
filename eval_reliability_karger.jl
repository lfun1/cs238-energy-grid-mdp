"""
Evaluate reliability of an energy grid graph with Karger's Algorithm for finding cost of minimum cut.

Author: Lisa Fung
Last Updated: 12/3/2024
"""

using Graphs
using GraphPlot         # for plotting graphs
using Compose, Cairo, Fontconfig    # for saving graphs


"""
Function to run Karger's Algorithm for many trials to achieve higher probability of success.
    Only applies to unweighted undirected graphs with at most 1 edge between vertices.
    Uses Graphs.karger_min_cut(g) to for each run of Karger's Algorithm.
"""
function karger_multiple_min_cut(g::Graph, num_trials::Int)
    min_cut_cost = nv(g) * (nv(g) - 1)   # Maximum cut value = maximum number of edges

    # Repeat Karger's algorithm num_trials times
    for i in 1:num_trials
        cut = karger_min_cut(g)
        cut_cost = karger_cut_cost(g, cut)
        min_cut_cost = min(min_cut_cost, cut_cost)
    end

    return min_cut_cost
end


"""
Create graphs and compute cost of minimum cut with Karger's Algorithm.
Note: Karger's Algorithm in karger_min_cut is only run once. 
Must run multiple times to get desired probability of success.

"""
function test_karger()
    # Create undirected graphs
    graphs = Vector{Graph}()
    push!(graphs, SimpleGraph(2))
    push!(graphs, path_graph(4))
    push!(graphs, complete_graph(5))

    karger_prob_fail = 0.01
    num_trials = 

    # Compute cost of minimum cut with Karger's Algorithm
    for (i, g) in enumerate(graphs)
        println("Graph ", i)
        println("Number of vertices: ", nv(g))
        println("Number of edges: ", ne(g))
        
        # min_cut = karger_min_cut(g)
        # println("Minimum cut: ", min_cut)
        # println("Minimum cut cost: ", karger_cut_cost(g, min_cut))
        num_trials = 
        println("Min cut cost (Karger's multiple times): ", karger_multiple_min_cut(g, 2))
    end

end

test_karger()