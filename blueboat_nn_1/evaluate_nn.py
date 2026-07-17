import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from config_pool import (BUOY_X, BUOY_Y, RADIUS, POOL_X_MIN, POOL_X_MAX, POOL_Y_MIN, POOL_Y_MAX, normalize_dataset_inputs)

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
trajectory = pd.read_csv("data/nn_trajectory.csv")

X = data[["x", "y", "cos_θ", "sin_θ"]].values.astype(np.float32)
Y = data[["u1", "u2"]].values.astype(np.float32)

X = normalize_dataset_inputs(X)

x_pos = data["x"].values
y_pos = data["y"].values

sin_theta = data["sin_θ"].values
cos_theta = data["cos_θ"].values
theta = np.arctan2(sin_theta, cos_theta)

X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

model = BoatNetwork()
model.load_state_dict(torch.load("boat_neural_network.pth"))
model.eval()

with torch.no_grad():
    Y_pred = model(X_tensor).numpy()

Y_true = Y

error_u1 = Y_pred[:, 0] - Y_true[:, 0]
error_u2 = Y_pred[:, 1] - Y_true[:, 1]

error_norm = np.sqrt(error_u1**2 + error_u2**2)

print("Mean error norm:", np.mean(error_norm))
print("Max error norm:", np.max(error_norm))
print("Mean absolute error u1:", np.mean(np.abs(error_u1)))
print("Mean absolute error u2:", np.mean(np.abs(error_u2)))

rho = np.sqrt((x_pos - BUOY_X)**2 + (y_pos - BUOY_Y)**2)
radius_error = rho - RADIUS

sum_error, x_edges, y_edges = np.histogram2d(x_pos, y_pos, bins=[30, 50], weights=error_norm)

count, _, _ = np.histogram2d(x_pos, y_pos, bins=[30, 50])

mean_error = sum_error / np.maximum(count, 1)

fig, ax = plt.subplots(figsize=(6, 8))

im = ax.imshow(mean_error.T, origin="lower", extent=[POOL_X_MIN, POOL_X_MAX, POOL_Y_MIN, POOL_Y_MAX], aspect="equal")

fig.colorbar(im, ax=ax, label="Mean prediction error")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Mean error by pool region")

ax.set_xlim(POOL_X_MIN, POOL_X_MAX)
ax.set_ylim(POOL_Y_MIN, POOL_Y_MAX)
ax.set_aspect("equal", adjustable="box")

# Desired circle
circle = plt.Circle((BUOY_X, BUOY_Y), RADIUS, fill=False, linestyle="--", linewidth=2, edgecolor="black", label="Desired radius")
ax.add_patch(circle)
ax.plot(trajectory["x"], trajectory["y"], color="black", linewidth=2, label="NN trajectory")
ax.plot(BUOY_X, BUOY_Y, "ko", markersize=8, label="Buoy")
ax.legend()

plt.show()