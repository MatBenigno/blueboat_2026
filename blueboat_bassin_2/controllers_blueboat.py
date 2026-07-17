#!/usr/bin/env python3
import numpy as np

from config_blueboat import (K_THETA, U0, U_MAX, K_D)


def sawtooth(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def carousel_field(mx, my, buoy, r):

    px = mx - buoy[0, 0]
    py = my - buoy[1, 0]

    R = np.sqrt(px**2 + py**2)

    if r < 1e-6:
        r = 1e-6

    theta = np.arctan2(py, px)

    X1 = (R / r) * np.cos(theta)
    X2 = (R / r) * np.sin(theta)

    Fx = -X1**3 - X1 * X2**2 + X1 - X2
    Fy = -X2**3 - X1**2 * X2 + X1 + X2

    return Fx, Fy


def control_vector_field(mx, my, theta, buoy, r, iz_data):

    Fx, Fy = carousel_field(mx, my, buoy, r)

    theta_d = np.arctan2(Fy, Fx)
    e_theta = sawtooth(theta_d - theta)

    u_diff = K_THETA * e_theta - K_D * iz_data

    u_left = U0 - u_diff
    u_right = U0 + u_diff

    # Saturation
    u_left = float(np.clip(u_left, -U_MAX, U_MAX))
    u_right = float(np.clip(u_right, -U_MAX, U_MAX))

    return u_left, u_right,  float(theta_d), float(e_theta)


