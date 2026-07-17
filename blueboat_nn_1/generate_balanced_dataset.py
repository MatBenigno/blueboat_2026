import numpy as np
import csv
from ddboatlib import sawtooth

from config_pool import (BUOY, RADIUS,DATASET_X_MIN, DATASET_X_MAX, DATASET_Y_MIN, DATASET_Y_MAX, N_X, N_Y, N_THETA, K_θ, U0)


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


def create_state(mx, my, θ):
    vx = 0
    vy = 0
    w = 0
    w1 = 1
    w2 = 1

    return np.array([[mx], [my], [θ], [vx], [vy], [w], [w1], [w2]])


def generate_balanced_dataset(filename):

    data = [] 

    buoy = BUOY
    r = RADIUS

    x_values = np.linspace(DATASET_X_MIN, DATASET_X_MAX, N_X)
    y_values = np.linspace(DATASET_Y_MIN, DATASET_Y_MAX, N_Y)
    θ_values = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)

    for mx in x_values:
        for my in y_values:
            for θ in θ_values:
                x = create_state(mx, my, θ)

                u = control_vector_field(x, buoy, r)

                u1 = u[0, 0]
                u2 = u[1, 0]

                nn_input = [
                    mx,
                    my,
                    np.cos(θ),
                    np.sin(θ)
                ]
                
                # Desired Output
                nn_output = [
                    u1,
                    u2
                ]

                data.append(nn_input + nn_output)


    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "sin_θ", "cos_θ", "u1", "u2"])
        writer.writerows(data)

    print(f"Dataset saved in {filename}")
    print(f"Number of examples: {len(data)}")

if __name__ == "__main__":
    generate_balanced_dataset("data/dataset_vector_field.csv")