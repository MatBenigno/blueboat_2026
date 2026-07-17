import pandas as pd
import matplotlib.pyplot as plt

from config_pool import RADIUS

manual_data = pd.read_csv("data/manual_trajectory.csv")
nn_data = pd.read_csv("data/nn_trajectory.csv")

fig, axs = plt.subplots(1, 2, figsize=(15, 8))

r = RADIUS

def setup_pool_axis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Desired circle around the buoy
    circle = plt.Circle((0, 0), r, fill=False, linestyle="--", linewidth=2, label="Desired radius")
    
    ax.add_patch(circle)
    ax.plot(0, 0, "ko", markersize=8, label="Buoy")
    ax.legend()

# Manual controller trajectory
axs[0].scatter(manual_data["x"], manual_data["y"], s=10)
setup_pool_axis(axs[0], "Vector Field Trajectory")

# Neural network trajectory
axs[1].scatter(nn_data["x"], nn_data["y"], s=10)
setup_pool_axis(axs[1], "Neural Network Trajectory")

plt.tight_layout()
plt.show()