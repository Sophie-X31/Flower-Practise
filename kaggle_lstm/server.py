import flwr as fl

def weighted_average(metrics):
    accuracies = [num_example * m["accuracy"] for num_example, m in metrics]
    examples = [num_example for num_example, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
    ),
)