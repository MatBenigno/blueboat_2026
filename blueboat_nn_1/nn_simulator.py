import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import csv

from ddboatlib import pool_to_latlong, init_figure, clear, draw_blueboat, ψ0
from config_pool import (BUOY, RADIUS, PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX, normalize_nn_input)

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


def f(x, u):
    x, u = x.flatten(), u.flatten()
    θ, vx, vy, w, w1, w2 = x[2], x[3], x[4], x[5], x[6], x[7]

    dx = vx*np.cos(θ) - vy*np.sin(θ)
    dy = vx*np.sin(θ) + vy*np.cos(θ)
    dθ = w

    dvx = w*vy - p5*vx*abs(vx) + p3*(w1*abs(w1) + w2*abs(w2))
    dvy = -w*vx - p6*vy*abs(vy)
    dw = -p7*w*abs(w) + p4*(w2*abs(w2) - w1*abs(w1))

    dw1 = -p1*w1*abs(w1) + p2*u[0]
    dw2 = -p1*w2*abs(w2) + p2*u[1]

    xdot = np.array([[dx], [dy], [dθ], [dvx], [dvy], [dw], [dw1], [dw2]])

    return xdot


def g(x):
    mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
    I = math.radians(64)

    lx, ly = pool_to_latlong(mx, my)

    m1 = np.sin(θ + ψ0) * np.cos(I)
    m2 = np.cos(θ + ψ0) * np.cos(I)
    m3 = -np.sin(I)

    return np.array([[lx], [ly], [m1], [m2], [m3], [w]])


def draw_buoy(ax):
    ax.plot(0, 0, 'ko', markersize=8)
    

def control_with_neural_network(x, model):
    mx = x[0, 0]
    my = x[1, 0]
    θ = x[2, 0]

    nn_input = normalize_nn_input(mx, my, θ)

    nn_input_tensor = torch.tensor(nn_input).unsqueeze(0)

    # Evaluate neural network without training
    with torch.no_grad():
        u_pred = model(nn_input_tensor)

    u_pred = u_pred.squeeze(0)

    u1 = u_pred[0].item()
    u2 = u_pred[1].item()

    # Saturation
    u1 = float(np.clip(u1, -1, 1))
    u2 = float(np.clip(u2, -1, 1))

    return np.array([[u1], [u2]])


# Model parameters
p1, p2, p3, p4, p5, p6, p7 = 0.07, 2200, 3.e-05, 15.e-05, 0.4, 5.0, 5.0


if __name__ == "__main__":

    # Load trained neural network
    model = BoatNetwork()
    model.load_state_dict(torch.load("boat_neural_network.pth"))
    model.eval()

    ax = init_figure(PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX)

    # Initial state
    x0, y0, θ0 = 10, 10, 0

    vx0, vy0, w0 = 0, 0, 0
    w10, w20 = 1, 1

    x = np.array([[x0, y0, θ0, vx0, vy0, w0, w10, w20]]).T

    # Buoy position
    buoy = BUOY
    r = RADIUS

    # Simulation parameters
    dt = 0.01
    tmax = 20

    trajectory = []

    for t in np.arange(0, tmax, dt):
        clear(ax)
        draw_buoy(ax)

        # Draw desired carousel circle
        circle = plt.Circle((BUOY[0, 0], BUOY[1, 0]), RADIUS, fill=False, linestyle="--")
        ax.add_patch(circle)

        u = 10 * control_with_neural_network(x, model)

        x = x + dt*f(x, u)
        y = g(x)

        mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
        d_buoy = np.sqrt((mx - BUOY[0, 0])**2 + (my - BUOY[1, 0])**2)

        trajectory.append([t, mx, my, θ, u[0,0], u[1,0], d_buoy])

        draw_blueboat(ax, mx, my, θ, 0.01*w1, 0.01*w2)

        plt.pause(0.001)

    with open("data/nn_trajectory.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "x", "y", "θ", "u1", "u2", "d_buoy"])
        writer.writerows(trajectory)

    print("Trajectory saved in data/nn_trajectory.csv")
