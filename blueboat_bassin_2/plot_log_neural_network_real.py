#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "log_neural_network_real.csv"
OUTPUT_DIR = "plots"

BUOY_X = 0.0
BUOY_Y = 0.0
TARGET_RADIUS = 3.0


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Columns found:")
print(df.columns.tolist())
print()
print("Number of samples:", len(df))

# Time relative to the beginning of the experiment
df["t"] = df["time"] - df["time"].iloc[0]

# Distance to the buoy
df["d_buoy"] = np.sqrt((df["x"] - BUOY_X)**2 + (df["y"] - BUOY_Y)**2)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Plot 1: trajectory x-y
# ============================================================

plt.figure(figsize=(8, 8))

plt.plot(df["x"], df["y"], label="Trajectory")
plt.scatter(df["x"].iloc[0], df["y"].iloc[0], marker="o", label="Start")
plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], marker="x", label="End")
plt.scatter(BUOY_X, BUOY_Y, marker="*", s=150, label="Buoy")

angle = np.linspace(0, 2*np.pi, 400)
circle_x = BUOY_X + TARGET_RADIUS * np.cos(angle)
circle_y = BUOY_Y + TARGET_RADIUS * np.sin(angle)
plt.plot(circle_x, circle_y, "--", label=f"Target radius = {TARGET_RADIUS} m")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Real trajectory - Neural network controller")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectory_xy.png", dpi=300)


# ============================================================
# Plot 2: distance to buoy
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["d_buoy"], label="Distance to buoy")
plt.axhline(TARGET_RADIUS, linestyle="--", label=f"Target radius = {TARGET_RADIUS} m")

plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.title("Distance to buoy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_to_buoy.png", dpi=300)


# ============================================================
# Plot 3: heading
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["yaw_attitude_deg"], label="Yaw from ATTITUDE [deg]")
plt.plot(df["t"], np.rad2deg(df["theta"]), label="Theta local [deg]")

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Boat heading")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heading.png", dpi=300)


# ============================================================
# Plot 4: neural network motor commands
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["u_left"], label="u_left")
plt.plot(df["t"], df["u_right"], label="u_right")

plt.xlabel("Time [s]")
plt.ylabel("Motor command")
plt.title("Neural network commands")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/motor_commands.png", dpi=300)


# ============================================================
# Plot 5: PWM sent to motors
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["u_left_pwm"], label="Left PWM")
plt.plot(df["t"], df["u_right_pwm"], label="Right PWM")
plt.axhline(1500, linestyle="--", label="Neutral PWM")

plt.xlabel("Time [s]")
plt.ylabel("PWM")
plt.title("PWM sent to thrusters")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/motor_pwm.png", dpi=300)


# ============================================================
# Plot 6: x and y over time
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["x"], label="x")
plt.plot(df["t"], df["y"], label="y")

plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Position over time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/position_time.png", dpi=300)


plt.show()

print()
print(f"Plots saved in folder: {OUTPUT_DIR}")