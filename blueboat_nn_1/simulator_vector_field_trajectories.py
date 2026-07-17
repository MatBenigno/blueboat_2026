import numpy as np
import math
import csv
import os

from config_pool import (BUOY, RADIUS, K_θ, U0)

def f(x, u):
    x, u = x.flatten(), u.flatten()
    θ, vx, vy, w, w1, w2 = x[2], x[3], x[4], x[5], x[6], x[7]

    dx = vx * np.cos(θ) - vy * np.sin(θ)
    dy = vx * np.sin(θ) + vy * np.cos(θ)
    dθ = w

    dvx = w * vy - p5 * vx * abs(vx) + p3 * (w1 * abs(w1) + w2 * abs(w2))
    dvy = -w * vx - p6 * vy * abs(vy)
    dw = -p7 * w * abs(w) + p4 * (w2 * abs(w2) - w1 * abs(w1))

    dw1 = -p1 * w1 * abs(w1) + p2 * u[0]
    dw2 = -p1 * w2 * abs(w2) + p2 * u[1]

    xdot = np.array([[dx],[dy],[dθ],[dvx],[dvy],[dw],[dw1],[dw2]])

    return xdot


def sawtooth(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def carousel_field_two(mx, my, buoy, r_d):
    px = mx - buoy[0, 0]
    py = my - buoy[1, 0]

    r = np.sqrt(px**2 + py**2)

    if r < 1e-6:
        r = 1e-6

    theta = np.arctan2(py, px)

    X1 = (r / r_d) * np.cos(theta)
    X2 = (r / r_d) * np.sin(theta)

    Fx = -X1**3 - X1 * X2**2 + X1 - X2
    Fy = -X2**3 - X1**2 * X2 + X1 + X2

    return Fx, Fy


def control_vector_field(x, buoy, r_d):
    mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])

    Fx, Fy = carousel_field_two(mx, my, buoy, r_d)

    θd = np.arctan2(Fy, Fx)
    eθ = sawtooth(θd - θ)

    k_θ = K_θ
    u0 = U0

    u_diff = k_θ * eθ

    u1 = u0 - u_diff
    u2 = u0 + u_diff

    u1 = np.clip(u1, -1, 1)
    u2 = np.clip(u2, -1, 1)

    return np.array([[u1], [u2]])


# Parameters
p1, p2, p3, p4, p5, p6, p7 = 0.07, 2200, 3.e-05, 15.e-05, 0.4, 5.0, 5.0

# Initial condition
x0, y0, θ0, vx0, vy0, w0, w10, w20 = -3, -7, 0, 0, 0, 0, 1, 1
x = np.array([[x0, y0, θ0, vx0, vy0, w0, w10, w20]]).T

# Buoy and desired radius
buoy = BUOY
r = RADIUS

# Simulation parameters
dt = 0.01
tmax = 50

trajectory = []

for t in np.arange(0, tmax, dt):

    u = 10 * control_vector_field(x, buoy, r)

    x = x + dt * f(x, u)

    mx, my, θ, vx, vy, w, w1, w2 = list(x[0:8, 0])
    d_buoy = np.sqrt((mx - BUOY[0, 0])**2 + (my - BUOY[1, 0])**2)

    trajectory.append([t,mx,my,θ,u[0, 0],u[1, 0],d_buoy])

os.makedirs("data", exist_ok=True)

with open("data/manual_trajectory.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["t", "x", "y", "θ", "u1", "u2", "d_buoy"])
    writer.writerows(trajectory)

print("Manual trajectory saved in data/manual_trajectory.csv")