import collections
from collections import OrderedDict

import flwr as fl
import torch

from centralized import load_data, load_model, train, test

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model 

net = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.net = model

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        self.net = set_parameters(self.net, parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters({}), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.net = set_parameters(self.net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader), {"accuracy": accuracy}

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(net).to_client(),
)