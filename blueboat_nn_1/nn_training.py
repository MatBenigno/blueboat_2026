import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time

from config_pool import normalize_dataset_inputs

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
 

data = pd.read_csv("data/dataset_vector_field.csv")

# Inputs
X = data[["x", "y", "cos_θ", "sin_θ"]].values.astype(np.float32)
X = normalize_dataset_inputs(X)

# Outputs
Y = data[["u1", "u2"]].values.astype(np.float32)

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

model = BoatNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10000

start_time = time.time()

for epoch in range(n_epochs):

    Y_pred = model(X_tensor)
    loss = criterion(Y_pred, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.6f}")


torch.save(model.state_dict(), "boat_neural_network.pth")

end_time = time.time()
training_time = end_time - start_time

print(f"Total training time: {training_time:.2f} seconds")
print(f"Total training time: {training_time/60:.2f} minutes")

print("Training finished.")
print("Model saved as boat_neural_network.pth")