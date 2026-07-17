#!/usr/bin/env python3
import time
import signal

from config_blueboat import (
    CONNECTION_STRING,
    SET_MANUAL_FROM_PYTHON,
    ARM_FROM_PYTHON,
    DEFAULT_MODE,
    RC_NEUTRAL,
    DRY_RUN,
)

from blueboat_mavlink import BlueBoatMavlink


running = True


# Start with a small value.
# If the motors do not move, try 80, then 100, then 120.
TEST_DELTA = 80

# Time for each test command
TEST_TIME_SEC = 2.0


def signal_handler(sig, frame):
    global running
    running = False


def send_neutral(boat, n=10):
    print("Sending neutral RC...")
    for _ in range(n):
        boat.send_rc_override(
            steering_pwm=RC_NEUTRAL,
            throttle_pwm=RC_NEUTRAL
        )
        time.sleep(0.1)


def send_rc_for_seconds(boat, steering_pwm, throttle_pwm, duration_sec):
    print(
        f"Sending RC | "
        f"CH1_steering={steering_pwm} "
        f"CH3_throttle={throttle_pwm} "
        f"for {duration_sec:.1f} s"
    )

    t_end = time.time() + duration_sec

    while running and time.time() < t_end:
        boat.send_rc_override(
            steering_pwm=steering_pwm,
            throttle_pwm=throttle_pwm
        )
        time.sleep(0.1)

    send_neutral(boat, n=5)


def wait_enter(message):
    input(message)


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("BlueBoat RC motor mixing test")
    print(f"Connection: {CONNECTION_STRING}")
    print(f"DRY_RUN = {DRY_RUN}")
    print("")
    print("This test uses RC Override:")
    print("  CH1 = steering / yaw")
    print("  CH3 = throttle")
    print("")
    print("It does not send direct servo PWM.")
    print("The autopilot mixes CH1 and CH3 into left/right motors.")
    print("")
    print(f"TEST_DELTA = {TEST_DELTA}")
    print("")

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
            return
    else:
        print("ARM_FROM_PYTHON = False")
        print("Arm the boat manually before pressing ENTER.")

    print("")
    print("Clearing previous RC override...")
    boat.stop_rc_override()

    print("")
    print("Sending neutral before tests...")
    send_neutral(boat, n=10)

    try:
        # ------------------------------------------------------------
        # Test both motors forward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test BOTH motors forward...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL,
            throttle_pwm=RC_NEUTRAL + TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        # ------------------------------------------------------------
        # Test both motors backward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test BOTH motors backward...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL,
            throttle_pwm=RC_NEUTRAL - TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        # ------------------------------------------------------------
        # Candidate: one motor forward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test MOTOR SIDE A forward candidate...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL + TEST_DELTA,
            throttle_pwm=RC_NEUTRAL + TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        # ------------------------------------------------------------
        # Candidate: other motor forward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test MOTOR SIDE B forward candidate...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL - TEST_DELTA,
            throttle_pwm=RC_NEUTRAL + TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        # ------------------------------------------------------------
        # Candidate: one motor backward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test MOTOR SIDE A backward candidate...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL - TEST_DELTA,
            throttle_pwm=RC_NEUTRAL - TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        # ------------------------------------------------------------
        # Candidate: other motor backward
        # ------------------------------------------------------------
        wait_enter("\nPress ENTER to test MOTOR SIDE B backward candidate...")
        send_rc_for_seconds(
            boat,
            steering_pwm=RC_NEUTRAL + TEST_DELTA,
            throttle_pwm=RC_NEUTRAL - TEST_DELTA,
            duration_sec=TEST_TIME_SEC
        )

        print("")
        print("Test sequence finished.")

    finally:
        print("")
        print("Stopping boat...")

        send_neutral(boat, n=10)

        print("Releasing RC override...")
        boat.release_rc_override()

        print("Disarming vehicle...")
        boat.disarm_vehicle()

        boat.set_mode(DEFAULT_MODE)

        print("Program finished.")


if __name__ == "__main__":
    main()