import networkx as nx

import decent_bench.utils.interoperability as iop
from decent_bench import benchmark
from decent_bench.agents import Agent
from decent_bench.costs import LinearRegressionCost
from decent_bench.distributed_algorithms import DGD
from decent_bench.networks import P2PNetwork

if __name__ == "__main__":
    iterations = int(globals().get("ITERATIONS", 10))
    n_agents = int(globals().get("N_AGENTS", 4))
    n_trials = int(globals().get("N_TRIALS", 1))

    iop.set_seed(7)

    costs, x_optimal, test_data = benchmark.create_regression_problem(cost_cls=LinearRegressionCost, n_agents=n_agents)
    agents = [Agent(agent_id=i, cost=cost) for i, cost in enumerate(costs)]
    network = P2PNetwork(graph=nx.complete_graph(agents))
    problem = benchmark.BenchmarkProblem(network=network, x_optimal=x_optimal, test_data=test_data)

    benchmark_result = benchmark.benchmark(
        algorithms=[DGD(iterations=iterations, step_size=0.01)],
        benchmark_problem=problem,
        n_trials=n_trials,
        max_processes=1,
    )
    metric_result = benchmark.compute_metrics(benchmark_result)
    benchmark.display_metrics(metric_result)
