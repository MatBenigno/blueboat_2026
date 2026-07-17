# config_pool.py
import numpy as np

# ============================================================
# Buoy reference
# ============================================================

BUOY_X = 0.0
BUOY_Y = 0.0
BUOY = np.array([[BUOY_X], [BUOY_Y]])

# Desired carousel radius
RADIUS = 3.0

# Gain
K_θ = 0.4
U0 = 0.3


# ============================================================
# Physical pool limits
# ============================================================

POOL_X_MIN = -10.0
POOL_X_MAX = 10.0

POOL_Y_MIN = -10.0
POOL_Y_MAX = 10.0


# ============================================================
# Dataset generation limits
# ============================================================

DATASET_X_MIN = -10.0
DATASET_X_MAX = 10.0

DATASET_Y_MIN = -10.0
DATASET_Y_MAX = 10.0

N_X = 80
N_Y = 80
N_THETA = 180


# ============================================================
# Plot limits
# ============================================================

PLOT_X_MIN = -10.0
PLOT_X_MAX = 10.0

PLOT_Y_MIN = -10.0
PLOT_Y_MAX = 10.0


# ============================================================
# Neural network input normalization
# ============================================================

NN_X_MIN = DATASET_X_MIN
NN_X_MAX = DATASET_X_MAX

NN_Y_MIN = DATASET_Y_MIN
NN_Y_MAX = DATASET_Y_MAX


# ============================================================
# Without normalization
# ============================================================

def normalize_value(value, value_min=None, value_max=None):
    return value

def normalize_xy(mx, my):
    return mx, my

def normalize_nn_input(mx, my, theta):

    return np.array([
        mx,
        my,
        np.cos(theta),
        np.sin(theta)
    ], dtype=np.float32)


def normalize_dataset_inputs(X):
    return X.astype(np.float32)


# ============================================================
# With normalization
# ============================================================

# def normalize_value(value, value_min, value_max):
#     """
#     Normalize one value to [-1, 1].

#     value_min -> -1
#     value_max ->  1
#     """

#     value_norm = 2.0 * (value - value_min) / (value_max - value_min) - 1.0

#     # Guarantees that the neural network input stays inside [-1, 1]
#     # value_norm = np.clip(value_norm, -1.0, 1.0)

#     return value_norm


# def normalize_xy(mx, my):
#     """
#     Normalize x and y position to [-1, 1].
#     """

#     x_norm = normalize_value(mx, NN_X_MIN, NN_X_MAX)
#     y_norm = normalize_value(my, NN_Y_MIN, NN_Y_MAX)

#     return x_norm, y_norm


# def normalize_nn_input(mx, my, theta):
#     """
#     Complete neural network input:
#     [x_norm, y_norm, sin(theta), cos(theta)]
#     """

#     x_norm, y_norm = normalize_xy(mx, my)

#     return np.array([
#         x_norm,
#         y_norm,
#         np.sin(theta),
#         np.cos(theta)
#     ], dtype=np.float32)


# def normalize_dataset_inputs(X):
#     """
#     Normalize a full dataset input matrix X.

#     X columns must be:
#     [x, y, sin_theta, cos_theta]
#     """

#     X_norm = X.copy()

#     X_norm[:, 0] = normalize_value(X_norm[:, 0], NN_X_MIN, NN_X_MAX)
#     X_norm[:, 1] = normalize_value(X_norm[:, 1], NN_Y_MIN, NN_Y_MAX)

#     # sin(theta) and cos(theta) are already in [-1, 1]
#     return X_norm.astype(np.float32)