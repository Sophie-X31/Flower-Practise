from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Wrap strategy, config into server
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
config = ServerConfig(num_rounds=3)
app = ServerApp(
    config=config,
    strategy=strategy,
)


if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )