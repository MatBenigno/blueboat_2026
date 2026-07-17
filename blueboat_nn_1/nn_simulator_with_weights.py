import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from ddboatlib import pool_to_latlong, init_figure, clear, draw_blueboat, ψ0
from config_pool import (BUOY, RADIUS, PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX, normalize_nn_input)

def f(x, u):
    '''
        θ = heading
        vx = velocity x
        vy = velocity y
        w = angular velocity
        w1 = propeller velocity 1
        w2 = propeller velocity 2
    '''
    x, u = x.flatten(), u.flatten()
    θ, vx, vy, w, w1, w2 = x[2], x[3], x[4], x[5], x[6], x[7]

    # Velocity x and y in the reference frame of the bassin
    dx = vx*np.cos(θ) - vy*np.sin(θ)
    dy = vx*np.sin(θ) + vy*np.cos(θ)
    dθ = w

    # Accelerations
    dvx = w*vy - p5*vx*abs(vx) + p3*(w1*abs(w1) + w2*abs(w2))
    dvy = -w*vx - p6*vy*abs(vy)
    dw = -p7*w*abs(w) + p4*(w2*abs(w2) - w1*abs(w1))

    # Propeller dynamics
    dw1 = -p1*w1*abs(w1) + p2*u[0]
    dw2 = -p1*w2*abs(w2) + p2*u[1]

    xdot = np.array([[dx], [dy], [dθ], [dvx], [dvy], [dw], [dw1], [dw2]])

    return xdot


def g(x):
    '''
        Simulated sensor output
    '''
    mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
    I = math.radians(64)

    lx, ly = pool_to_latlong(mx, my)

    m1 = np.sin(θ + ψ0) * np.cos(I)
    m2 = np.cos(θ + ψ0) * np.cos(I)
    m3 = -np.sin(I)

    return np.array([[lx], [ly], [m1], [m2], [m3], [w]])


def draw_buoy(ax):
    ax.plot(0, 0, 'ko', markersize=8)



WEIGHTS_FILE = "data/neural_network_weights.csv"


def relu(x):
    return np.maximum(0.0, x)


def load_weights_from_csv(filename):
    """
    Load neural network weights from a CSV file.

    Expected architecture:
    input  : 4
    hidden1: 8
    hidden2: 8
    output : 2

    Expected total number of weights:
    4*8 + 8*8 + 8*2 = 112
    """

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        row = next(reader)

    weights = np.array([float(value) for value in row], dtype=np.float32)

    if len(weights) != 112:
        raise ValueError(f"Expected 112 weights, got {len(weights)}")

    idx = 0

    W1 = weights[idx:idx + 4*8].reshape(4, 8)
    idx += 4*8

    W2 = weights[idx:idx + 8*8].reshape(8, 8)
    idx += 8*8

    W3 = weights[idx:idx + 8*2].reshape(8, 2)
    idx += 8*2

    return W1, W2, W3


def neural_network_forward(nn_input, weights):
    """
    Forward pass using only NumPy.

    nn_input must have shape (4,):
    [x, y, sin(theta), cos(theta)]
    """

    W1, W2, W3 = weights

    z1 = nn_input @ W1
    h1 = relu(z1)

    z2 = h1 @ W2
    h2 = relu(z2)

    output = h2 @ W3

    return output

    

def control_with_neural_network(x, weights):
    mx = x[0, 0]
    my = x[1, 0]
    θ = x[2, 0]

    nn_input = normalize_nn_input(mx, my, θ)

    u_pred = neural_network_forward(nn_input, weights)

    u1 = float(u_pred[0])
    u2 = float(u_pred[1])

    # Saturation
    u1 = float(np.clip(u1, -1, 1))
    u2 = float(np.clip(u2, -1, 1))

    return np.array([[u1], [u2]])



# Model parameters
p1, p2, p3, p4, p5, p6, p7 = 0.07, 2200, 3.e-05, 15.e-05, 0.4, 5.0, 5.0


if __name__ == "__main__":

    # Load neural network weights from CSV
    weights = load_weights_from_csv(WEIGHTS_FILE)
    print("Neural network weights loaded from CSV.")

    ax = init_figure(PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX)

    # Initial state
    x0, y0, θ0 = 100, 100, 0

    vx0, vy0, w0 = 0, 0, 0
    w10, w20 = 1, 1

    x = np.array([[x0, y0, θ0, vx0, vy0, w0, w10, w20]]).T

    dt = 0.01
    tmax = 20

    trajectory = []

    for t in np.arange(0, tmax, dt):
        clear(ax)
        draw_buoy(ax)

        circle = plt.Circle((BUOY[0, 0], BUOY[1, 0]), RADIUS, fill=False, linestyle="--")
        ax.add_patch(circle)

        # Neural network controller from CSV weights
        u = control_with_neural_network(x, weights)

        x = x + dt*f(x, u)
        y = g(x)

        mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
        d_buoy = np.sqrt((mx - BUOY[0, 0])**2 + (my - BUOY[1, 0])**2)

        trajectory.append([t, mx, my, θ, u[0, 0], u[1, 0], d_buoy])

        draw_blueboat(ax, mx, my, θ, 0.01*w1, 0.01*w2)

        plt.pause(0.001)

    with open("data/nn_trajectory.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "x", "y", "θ", "u1", "u2", "d_buoy"])
        writer.writerows(trajectory)

    print("Trajectory saved in data/nn_trajectory.csv")