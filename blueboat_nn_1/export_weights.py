import csv
import torch
import torch.nn as nn


class BoatNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 2, bias=False)
        )

    def forward(self, x):
        return self.net(x)

model = BoatNetwork()
model.load_state_dict(torch.load("boat_neural_network.pth"))
model.eval()

weights = []

linear_layers = [model.net[0], model.net[2], model.net[4]]

for layer in linear_layers:
    W = layer.weight.detach().numpy().T

    for row in W:
        for value in row:
            weights.append(float(value))

print("Number of exported weights:", len(weights))

if len(weights) != 112:
    raise ValueError(f"Expected 112 weights, got {len(weights)}")

with open("data/neural_network_weights.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(weights)

print("Weights saved in data/neural_network_weights.csv")