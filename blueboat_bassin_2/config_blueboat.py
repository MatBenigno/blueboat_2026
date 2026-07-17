#!/usr/bin/env python3
import math
import numpy as np

# ============================================================
# MAVLink connection
# ============================================================

CONNECTION_STRING = "udpin:192.168.2.10:14550"

# Control loop frequency
CONTROL_HZ = 15.0


# ============================================================
# GPS reference point
# ============================================================

LAT0_DEG = 48.3154688000
LON0_DEG = -4.2826982000

X0_LOCAL = 0.0
Y0_LOCAL = 0.0

# Rotation of the local x-y frame relative to East/North.
PSI0_RAD = math.radians(356.12 - 360.0)


# ============================================================
# Carousel parameters
# ============================================================

BUOY_XY = np.array([[0.0], [0.0]])
R_CAROUSEL = 5.0


# ============================================================
# Vector field controller parameters
# ============================================================

K_THETA = 0.6
U0 = 0.3
K_D = 0.0

U_MAX = 1.0


# ============================================================
# GPS / sensor safety (data)
# ============================================================

DATA_TIMEOUT_SEC = 3.0

# ============================================================
# Arming / mode parameters
# ============================================================

SET_MANUAL_FROM_PYTHON = True

ARM_FROM_PYTHON = True

DISARM_ON_EXIT = True

DEFAULT_MODE = "MANUAL"


# ============================================================
# RC Override parameters
# ============================================================

RC_NEUTRAL = 1500

DEADZONE = 0.08

