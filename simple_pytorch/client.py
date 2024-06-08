from collections import OrderedDict
from flwr.client import NumPyClient, ClientApp, start_client
import torch
from centralized import train, test, load_data, load_model


net = load_model()
trainloader, testloader = load_data()

class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )