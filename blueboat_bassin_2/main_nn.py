#!/usr/bin/env python3
import time
import math
import signal
import csv

from config_blueboat import (CONTROL_HZ, SET_MANUAL_FROM_PYTHON, ARM_FROM_PYTHON, DEFAULT_MODE, DATA_TIMEOUT_SEC)
from blueboat_mavlink import BlueBoatMavlink
from coordinates_blueboat import gps_to_xy, yaw_attitude_rad_to_theta_local
from neural_network_blueboat import (load_neural_network, control_with_neural_network_debug)


def command_to_pwm_with_min(cmd, neutral=1500, pwm_min=1250, pwm_max=1750, u_max=1.0, min_delta=0):

    cmd = max(-u_max, min(u_max, float(cmd)))

    if abs(cmd) < 1e-6:
        return neutral

    delta_max = pwm_max - neutral
    pwm_delta = (cmd / u_max) * delta_max

    if min_delta > 0 and abs(pwm_delta) < min_delta:
        pwm_delta = math.copysign(min_delta, pwm_delta)

    pwm = neutral + pwm_delta
    pwm = max(pwm_min, min(pwm_max, pwm))

    return int(pwm)


def send_neutral_rc(boat):
    boat.send_rc_override(steering_pwm=1500, throttle_pwm=1500)


running = True

def signal_handler(sig, frame):
    global running
    running = False


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting BlueBoat neural network controller")
    print(f"Control frequency = {CONTROL_HZ} Hz")

    weights = load_neural_network("data/weights_antoine_leboeuf.csv")
    print("Neural network weights loaded.")

    boat = BlueBoatMavlink()

    if SET_MANUAL_FROM_PYTHON:
        mode_ok = boat.set_mode(DEFAULT_MODE)

        if not mode_ok:
            print(f"Could not set {DEFAULT_MODE} mode.")
            return

    if ARM_FROM_PYTHON:
        armed_ok = boat.arm_vehicle()

        if not armed_ok:
            print("Could not arm the vehicle.")
            print("Check arming messages in BlueOS/Cockpit.")
            return
    else:
        print("")
        print("ARM_FROM_PYTHON = False")
        print("Arm the boat manually in BlueOS/Cockpit.")
        print("")

    print("Clearing previous RC override...")
    boat.stop_rc_override()

    print("Sending neutral RC command before starting controller...")
    for _ in range(5):
        send_neutral_rc(boat)
        time.sleep(0.1)

    boat.configure_streams()

    dt = 1.0 / CONTROL_HZ

    log_filename = "log_neural_network_real.csv"

    with open(log_filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["time","lat_deg","lon_deg","yaw_attitude_deg","x","y","theta","sin_theta","cos_theta","u_left","u_right","u_left_pwm","u_right_pwm"])

        try:
            last_gps_data = None
            last_yaw_data = None

            last_gps_time = time.monotonic()
            last_yaw_time = time.monotonic()

            while running:
                t0 = time.time()

                gps_data = boat.read_gps_position(timeout=1.0)
                yaw_data = boat.read_attitude_yaw(timeout=1.0)

                now = time.monotonic()

                if gps_data is not None:
                    last_gps_data = gps_data
                    last_gps_time = now

                if yaw_data is not None:
                    last_yaw_data = yaw_data
                    last_yaw_time = now

                gps_valid = (last_gps_data is not None and now - last_gps_time <= DATA_TIMEOUT_SEC)

                yaw_valid = (last_yaw_data is not None and now - last_yaw_time <= DATA_TIMEOUT_SEC)

                if not gps_valid or not yaw_valid:
                    print("GPS or yaw not valid. Sending neutral RC.")
                    send_neutral_rc(boat)
                    time.sleep(dt)
                    continue

                lat_deg, lon_deg = last_gps_data
                yaw_rad = last_yaw_data

                mx, my = gps_to_xy(lat_deg, lon_deg)
                theta = yaw_attitude_rad_to_theta_local(yaw_rad)

                u_left, u_right, nn_input = control_with_neural_network_debug(mx, my, theta, weights)

                u_throttle = (u_left + u_right) / 2.0
                u_steering = (u_left - u_right) / 2.0

                throttle_pwm = command_to_pwm_with_min(u_throttle, neutral=1500, pwm_min=1250, pwm_max=1750, u_max=1.0, min_delta=0)

                steering_pwm = command_to_pwm_with_min(u_steering, neutral=1500, pwm_min=1250, pwm_max=1750, u_max=1.0, min_delta=0)

                boat.send_rc_override(steering_pwm=steering_pwm, throttle_pwm=throttle_pwm)

                u_left_pwm = command_to_pwm_with_min(u_left, neutral=1500, pwm_min=1250, pwm_max=1750, u_max=1.0, min_delta=0)

                u_right_pwm = command_to_pwm_with_min(u_right, neutral=1500, pwm_min=1250, pwm_max=1750, u_max=1.0, min_delta=0)

                sin_theta = nn_input[2]
                cos_theta = nn_input[3]

                print(
                    f"x={mx:+7.2f} m |"
                    f"y={my:+7.2f} m |"
                    f"yaw={math.degrees(yaw_rad):7.2f} deg |"
                    f"theta={math.degrees(theta):+7.2f} deg |"
                    f"input=[{nn_input[0]:+.2f}, {nn_input[1]:+.2f}, {nn_input[2]:+.2f}, {nn_input[3]:+.2f}] |"
                    f"u=({u_left:+5.2f}, {u_right:+5.2f}) |"
                    f"pwm=({u_left_pwm:+5.2f}, {u_right_pwm:+5.2f}) |"
                )

                writer.writerow([time.time(),lat_deg,lon_deg,math.degrees(yaw_rad),mx,my,theta,sin_theta,cos_theta,u_left,u_right,u_left_pwm,u_right_pwm])
                
                file.flush()
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))

        finally:
            print("")
            print("Stopping boat...")

            for _ in range(5):
                send_neutral_rc(boat)
                time.sleep(0.1)
                
            boat.stop_rc_override()
            boat.disarm_vehicle()
            boat.set_mode(DEFAULT_MODE)

            print("Program finished.")
            print(f"Log saved in {log_filename}")

if __name__ == "__main__":
    main()