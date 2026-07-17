#!/usr/bin/env python3
import csv
import numpy as np

from config_blueboat import U_MAX


# ============================================================
# Neural network architecture (WITHOUT BIAS)
# ============================================================

INPUT_SIZE = 4
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 8
OUTPUT_SIZE = 2

EXPECTED_NUMBER_OF_WEIGHTS = (
    INPUT_SIZE * HIDDEN_SIZE_1 +
    HIDDEN_SIZE_1 * HIDDEN_SIZE_2 +
    HIDDEN_SIZE_2 * OUTPUT_SIZE
)

DEFAULT_WEIGHTS_PATH = "data/weights_antoine_leboeuf.csv"


def relu(x):
    return np.maximum(0.0, x)


def load_neural_network(weights_path=DEFAULT_WEIGHTS_PATH):

    # Archive CSV in row
    with open(weights_path, "r", newline="") as file:
        reader = csv.reader(file)
        row = next(reader)

    weights_list = np.array(
        [float(value) for value in row],
        dtype=np.float32
    )

    # Archive CSV in column
    # with open(weights_path, "r", newline="") as file:
    #     reader = csv.reader(file)

    #     values = []
    #     for row in reader:
    #         for value in row:
    #             value = value.strip()
    #             if value != "":
    #                 values.append(float(value))

    # weights_list = np.array(values, dtype=np.float32)


    if len(weights_list) != EXPECTED_NUMBER_OF_WEIGHTS:
        raise ValueError(
            f"Expected {EXPECTED_NUMBER_OF_WEIGHTS} weights, "
            f"got {len(weights_list)}"
        )

    idx = 0

    # Layer 1: 4 -> 8
    W1 = weights_list[idx:idx + INPUT_SIZE * HIDDEN_SIZE_1]
    W1 = W1.reshape(INPUT_SIZE, HIDDEN_SIZE_1)
    idx += INPUT_SIZE * HIDDEN_SIZE_1

    # Layer 2: 8 -> 8
    W2 = weights_list[idx:idx + HIDDEN_SIZE_1 * HIDDEN_SIZE_2]
    W2 = W2.reshape(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
    idx += HIDDEN_SIZE_1 * HIDDEN_SIZE_2

    # Layer 3: 8 -> 2
    W3 = weights_list[idx:idx + HIDDEN_SIZE_2 * OUTPUT_SIZE]
    W3 = W3.reshape(HIDDEN_SIZE_2, OUTPUT_SIZE)

    print(f"Neural network weights loaded from {weights_path}")
    print(f"Number of weights: {len(weights_list)}")

    return W1, W2, W3


def neural_network_forward(nn_input, weights):

    W1, W2, W3 = weights

    h1 = relu(nn_input @ W1)
    h2 = relu(h1 @ W2)
    output = h2 @ W3

    return output


def control_with_neural_network(mx, my, theta, weights):

    nn_input = np.array(
        [
            mx,
            my,
            np.cos(theta),
            np.sin(theta)
        ],
        dtype=np.float32
    )

    u_pred = neural_network_forward(nn_input, weights)

    u_left = float(np.clip(u_pred[0], -U_MAX, U_MAX))
    u_right = float(np.clip(u_pred[1], -U_MAX, U_MAX))

    return u_left, u_right


def control_with_neural_network_debug(mx, my, theta, weights):

    nn_input = np.array(
        [
            mx,
            my,
            np.cos(theta),
            np.sin(theta)
        ],
        dtype=np.float32
    )

    u_pred = neural_network_forward(nn_input, weights)

    u_left = float(np.clip(u_pred[0], -U_MAX, U_MAX))
    u_right = float(np.clip(u_pred[1], -U_MAX, U_MAX))

    return u_left, u_right, nn_input


# ============================================================
# Neural network architecture (WITH BIAS)
# ============================================================

# INPUT_SIZE = 4
# HIDDEN_SIZE_1 = 8
# HIDDEN_SIZE_2 = 8
# OUTPUT_SIZE = 2

# EXPECTED_NUMBER_OF_WEIGHTS = (
#     INPUT_SIZE * HIDDEN_SIZE_1 +
#     HIDDEN_SIZE_1 +
#     HIDDEN_SIZE_1 * HIDDEN_SIZE_2 +
#     HIDDEN_SIZE_2 +
#     HIDDEN_SIZE_2 * OUTPUT_SIZE +
#     OUTPUT_SIZE
# )

# DEFAULT_WEIGHTS_PATH = "data/weights.csv"


# def relu(x):
#     return np.maximum(0.0, x)


# def load_neural_network(weights_path=DEFAULT_WEIGHTS_PATH):

#     with open(weights_path, "r", newline="") as file:
#         reader = csv.reader(file)

#         values = []
#         for row in reader:
#             for value in row:
#                 value = value.strip()
#                 if value != "":
#                     values.append(float(value))

#     weights_list = np.array(values, dtype=np.float32)

#     if len(weights_list) != EXPECTED_NUMBER_OF_WEIGHTS:
#         raise ValueError(
#             f"Expected {EXPECTED_NUMBER_OF_WEIGHTS} weights, "
#             f"got {len(weights_list)}"
#         )

#     idx = 0

#     W1 = weights_list[idx:idx + INPUT_SIZE * HIDDEN_SIZE_1]
#     W1 = W1.reshape(INPUT_SIZE, HIDDEN_SIZE_1)
#     idx += INPUT_SIZE * HIDDEN_SIZE_1

#     b1 = weights_list[idx:idx + HIDDEN_SIZE_1]
#     idx += HIDDEN_SIZE_1

#     W2 = weights_list[idx:idx + HIDDEN_SIZE_1 * HIDDEN_SIZE_2]
#     W2 = W2.reshape(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
#     idx += HIDDEN_SIZE_1 * HIDDEN_SIZE_2

#     b2 = weights_list[idx:idx + HIDDEN_SIZE_2]
#     idx += HIDDEN_SIZE_2

#     W3 = weights_list[idx:idx + HIDDEN_SIZE_2 * OUTPUT_SIZE]
#     W3 = W3.reshape(HIDDEN_SIZE_2, OUTPUT_SIZE)
#     idx += HIDDEN_SIZE_2 * OUTPUT_SIZE

#     b3 = weights_list[idx:idx + OUTPUT_SIZE]
#     idx += OUTPUT_SIZE

#     print(f"Neural network weights loaded from {weights_path}")
#     print(f"Number of parameters: {len(weights_list)}")

#     return W1, b1, W2, b2, W3, b3


# def neural_network_forward(nn_input, weights):

#     W1, b1, W2, b2, W3, b3 = weights

#     h1 = relu(nn_input @ W1 + b1)
#     h2 = relu(h1 @ W2 + b2)
#     output = h2 @ W3 + b3

#     return output


# def control_with_neural_network(mx, my, theta, weights):

#     nn_input = np.array(
#         [
#             mx,
#             my,
#             np.cos(theta),
#             np.sin(theta)
#         ],
#         dtype=np.float32
#     )

#     u_pred = neural_network_forward(nn_input, weights)

#     u_left = float(np.clip(u_pred[0], -U_MAX, U_MAX))
#     u_right = float(np.clip(u_pred[1], -U_MAX, U_MAX))

#     return u_left, u_right


# def control_with_neural_network_debug(mx, my, theta, weights):

#     nn_input = np.array(
#         [
#             mx,
#             my,
#             np.cos(theta),
#             np.sin(theta)
#         ],
#         dtype=np.float32
#     )

#     u_pred = neural_network_forward(nn_input, weights)

#     u_left = float(np.clip(u_pred[0], -U_MAX, U_MAX))
#     u_right = float(np.clip(u_pred[1], -U_MAX, U_MAX))

#     return u_left, u_right, nn_input