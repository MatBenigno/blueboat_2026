#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "log_vector_field_real.csv"
OUTPUT_DIR = "plots_vector_field"

BUOY_X = 0.0
BUOY_Y = 0.0
TARGET_RADIUS = 5.0


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Columns found:")
print(df.columns.tolist())
print()
print("Number of samples:", len(df))

# Tempo começando em zero
df["t"] = df["time"] - df["time"].iloc[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Plot 1: trajectory x-y
# ============================================================

plt.figure(figsize=(8, 8))

plt.plot(df["x"], df["y"], label="Real trajectory", linewidth=2)

plt.scatter(df["x"].iloc[0], df["y"].iloc[0], marker="o", s=80, label="Start")
plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], marker="x", s=100, label="End")
plt.scatter(BUOY_X, BUOY_Y, marker="*", s=180, label="Buoy")

# Desired circle
angle = np.linspace(0, 2*np.pi, 500)
circle_x = BUOY_X + TARGET_RADIUS * np.cos(angle)
circle_y = BUOY_Y + TARGET_RADIUS * np.sin(angle)

plt.plot(
    circle_x,
    circle_y,
    "--",
    linewidth=2,
    label=f"Target radius = {TARGET_RADIUS} m"
)

# Heading arrows along the trajectory
step = max(1, len(df) // 25)

plt.quiver(
    df["x"].iloc[::step],
    df["y"].iloc[::step],
    np.cos(df["theta"].iloc[::step]),
    np.sin(df["theta"].iloc[::step]),
    angles="xy",
    scale_units="xy",
    scale=2.5,
    width=0.004,
    label="Boat heading"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Real trajectory - Vector field controller")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trajectory_xy_vector_field.png", dpi=300)


# ============================================================
# Plot 2: distance to buoy
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["distance_to_buoy"], label="Distance to buoy")
plt.axhline(TARGET_RADIUS, linestyle="--", label=f"Target radius = {TARGET_RADIUS} m")

plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.title("Distance to buoy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/distance_to_buoy_vector_field.png", dpi=300)


# ============================================================
# Plot 3: radial error
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["radial_error"], label="Radial error")
plt.axhline(0.0, linestyle="--", label="Desired radial error = 0")

plt.xlabel("Time [s]")
plt.ylabel("Radial error [m]")
plt.title("Radial error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radial_error_vector_field.png", dpi=300)


# ============================================================
# Plot 4: heading and desired heading
# ============================================================

plt.figure(figsize=(10, 5))

theta_deg = np.rad2deg(df["theta"])
theta_d_deg = np.rad2deg(df["theta_d"])

plt.plot(df["t"], theta_deg, label="theta - boat heading")
plt.plot(df["t"], theta_d_deg, label="theta_d - desired heading")

plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Heading vs desired heading")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heading_vs_desired_vector_field.png", dpi=300)


# ============================================================
# Plot 5: angular error
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], np.rad2deg(df["e_theta"]), label="e_theta")
plt.axhline(0.0, linestyle="--", label="Desired angular error = 0")

plt.xlabel("Time [s]")
plt.ylabel("Angular error [deg]")
plt.title("Heading error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/angular_error_vector_field.png", dpi=300)


# ============================================================
# Plot 6: control commands
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(df["t"], df["u_left"], label="u_left")
plt.plot(df["t"], df["u_right"], label="u_right")

plt.xlabel("Time [s]")
plt.ylabel("Motor command")
plt.title("Vector field motor commands")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/motor_commands_vector_field.png", dpi=300)


# ============================================================
# Plot 7: PWM sent to motors
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
plt.savefig(f"{OUTPUT_DIR}/motor_pwm_vector_field.png", dpi=300)


plt.show()

print()
print(f"Plots saved in folder: {OUTPUT_DIR}")