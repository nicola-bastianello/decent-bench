import networkx as nx

from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.costs import LinearRegressionCost
from decent_bench.networks import P2PNetwork
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate

if __name__ == "__main__":
    n_agents = int(globals().get("N_AGENTS", 6))
    costs, x_optimal, test_data = benchmark.create_regression_problem(cost_cls=LinearRegressionCost, n_agents=n_agents)

    agents = [
        Agent(agent_id=i, cost=cost, activation=UniformActivationRate(0.8))
        for i, cost in enumerate(costs)
    ]
    graph = nx.cycle_graph(agents)

    network = P2PNetwork(
        graph=graph,
        message_compression=Quantization(4),
        message_noise=GaussianNoise(0.0, 0.001),
        message_drop=UniformDropRate(0.05),
    )
    problem = benchmark.BenchmarkProblem(network=network, x_optimal=x_optimal, test_data=test_data)
