"""
Plot graphs
"""

using GraphPlot         # for plotting graphs
using Compose, Cairo, Fontconfig    # for saving graphs

# Experiment Initial results
experiments_initial = Dict{Any, Any}((2.0, 50) => Dict{Symbol, Any}(:max_degree => 1, :total_cost => 24, :discounted_reward => 45.428202133655056, :total_steps => 0, :final_graph => SimpleGraph{Int64}(1, [Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], [10], Int64[], [8], Int64[], Int64[], Int64[], Int64[]]), :total_reward => 149.0, :reliability => 0), (5.0, 10) => Dict{Symbol, Any}(:max_degree => 1, :total_cost => 25, :discounted_reward => 38.11413417866016, :total_steps => 6, :final_graph => SimpleGraph{Int64}(1, [[2], [1]]), :total_reward => 112.0, :reliability => 1), (8.0, 20) => Dict{Symbol, Any}(:max_degree => 1, :total_cost => 31, :discounted_reward => 63.26233454094308, :total_steps => 10, :final_graph => SimpleGraph{Int64}(1, [[2], [1]]), :total_reward => 173.0, :reliability => 1), (2.0, 10) => Dict{Symbol, Any}(:max_degree => 0, :total_cost => 20, :discounted_reward => 38.45760313375447, :total_steps => 10, :final_graph => SimpleGraph{Int64}(0, [Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[]]), :total_reward => 118.0, :reliability => 0), (5.0, 20) => Dict{Symbol, Any}(:max_degree => 2, :total_cost => 28, :discounted_reward => 30.440225077754462, :total_steps => 14, :final_graph => SimpleGraph{Int64}(2, [[2, 3], [1], [1]]), :total_reward => 104.0, :reliability => 1), (8.0, 50) => Dict{Symbol, Any}(:max_degree => 1, :total_cost => 30, :discounted_reward => 63.26233454094308, :total_steps => 0, :final_graph => SimpleGraph{Int64}(1, [[2], [1]]), :total_reward => 173.0, :reliability => 1), (2.0, 20) => Dict{Symbol, Any}(:max_degree => 0, :total_cost => 22, :discounted_reward => 37.81956597056589, :total_steps => 18, :final_graph => SimpleGraph{Int64}(0, [Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[], Int64[]]), :total_reward => 126.0, :reliability => 0), (8.0, 10) => Dict{Symbol, Any}(:max_degree => 1, :total_cost => 24, :discounted_reward => 44.36251454094308, :total_steps => 7, :final_graph => SimpleGraph{Int64}(1, [[2], [1]]), :total_reward => 145.0, :reliability => 1), (5.0, 50) => Dict{Symbol, Any}(:max_degree => 2, :total_cost => 42, :discounted_reward => 57.38177985322602, :total_steps => 0, :final_graph => SimpleGraph{Int64}(3, [[2, 3], [1, 3], [1, 2]]), :total_reward => 181.0, :reliability => 2))

alpha_r_list = [2.0, 5.0, 8.0]
budget_list = [10, 20, 50]

for alpha_r in alpha_r_list
    for budget in budget_list
        println("Experiment: alpha_r: ", alpha_r, ", budget: ", budget)
        println(experiments_initial[(alpha_r, budget)])
        println()
        # Plot each graph
        # plot = gplot(experiments_initial[(alpha_r, budget)][:final_graph], layout=circular_layout)
        # draw(PDF(string("./experiment_initial/graph_alpha_r_", alpha_r, "_budget_", budget, ".pdf"), 16cm, 16cm), plot)
    end
end

# write_gph(K2_G_best, node_names, string("project1/outputs_", dataset, "/K2_best_G_order_random.gph"))