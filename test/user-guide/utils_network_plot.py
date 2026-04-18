import networkx as nx

from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.costs import LinearRegressionCost
from decent_bench.networks import P2PNetwork
from decent_bench.utils import network_utils

if __name__ == "__main__":
    n_agents = int(globals().get("N_AGENTS", 6))
    costs, _, _ = benchmark.create_regression_problem(cost_cls=LinearRegressionCost, n_agents=n_agents)
    agents = [Agent(agent_id=i, cost=cost) for i, cost in enumerate(costs)]
    problem_network = P2PNetwork(graph=nx.cycle_graph(agents))

    ax = network_utils.plot_network(problem_network.graph, layout="circular", with_labels=True)
