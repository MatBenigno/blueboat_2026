#!/usr/bin/env python3
import time
import numpy as np

from pymavlink import mavutil
from config_blueboat import (CONNECTION_STRING, CONTROL_HZ, RC_NEUTRAL)

class BlueBoatMavlink:

    def __init__(self, connection_string=CONNECTION_STRING):
        print(f"Connecting to MAVLink: {connection_string}")

        self.master = mavutil.mavlink_connection(connection_string)

        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()

        print(
            f"Heartbeat received: "
            f"system={self.master.target_system}, "
            f"component={self.master.target_component}"
        )

        self.last_gps = None
        self.last_attitude = None

        self.last_gps_time = 0.0
        self.last_attitude_time = 0.0


    def set_message_interval(self, message_id, hz):

        interval_us = int(1e6 / hz)

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0
        )


    def configure_streams(self):

        self.set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            CONTROL_HZ
        )

        self.set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            CONTROL_HZ
        )

        self.set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_AHRS,
            CONTROL_HZ
        )


    def read_gps_position(self, timeout=1.0):

        msg = self.master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=timeout
        )

        if msg is None:
            return None

        lat_deg = msg.lat / 1e7
        lon_deg = msg.lon / 1e7

        self.last_gps = (lat_deg, lon_deg)
        self.last_gps_time = time.monotonic()

        return self.last_gps
        

    def read_attitude_yaw(self, timeout=1.0):

        msg = self.master.recv_match(
            type="ATTITUDE",
            blocking=True,
            timeout=timeout
        )

        if msg is None:
            return None

        yaw_rad = float(msg.yaw)

        self.last_attitude = yaw_rad
        self.last_attitude_time = time.monotonic()

        return yaw_rad
    

    def read_ahrs_iz(self, timeout=1.0):

        msg = self.master.recv_match(
            type="AHRS",
            blocking=True,
            timeout=timeout
        )

        if msg is None:
            return 0

        
        iz = float(msg.omegaIz)

        if iz is not None:
            return iz
        else: 
            return 0
        
    
    def set_mode(self, mode_name):

        mode_mapping = self.master.mode_mapping()

        if mode_mapping is None:
            print("Mode mapping is not available.")
            return False

        if mode_name not in mode_mapping:
            print(f"Mode {mode_name} is not available.")
            print("Available modes:", list(mode_mapping.keys()))
            return False

        mode_id = mode_mapping[mode_name]

        print(f"Setting mode to {mode_name}...")
        self.master.set_mode(mode_id)

        time.sleep(1.0)

        return True


    def arm_vehicle(self):

        print("Sending arm command...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # 1 = arm
            0,
            0,
            0,
            0,
            0,
            0
        )

        print("Waiting for vehicle to arm...")

        try:
            self.master.motors_armed_wait()
            print("Vehicle armed.")
            return True

        except Exception as e:
            print(f"Arming failed: {e}")
            return False


    def disarm_vehicle(self):

        print("Sending disarm command...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # 0 = disarm
            0,
            0,
            0,
            0,
            0,
            0
        )

        time.sleep(1.0)

        print("Disarm command sent.")
        return True
    

    def send_rc_override(self, steering_pwm, throttle_pwm):

        steering_pwm = int(steering_pwm)
        throttle_pwm = int(throttle_pwm)

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            steering_pwm,   # CH1 steering / yaw
            0,              # CH2 unchanged
            throttle_pwm,   # CH3 throttle
            0,              # CH4 unchanged
            0, 0, 0, 0
        )


    def release_rc_override(self):

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )


    def stop_rc_override(self, n=10):

        print("Stopping RC override...")

        for _ in range(n):
            self.send_rc_override(
                steering_pwm=RC_NEUTRAL,
                throttle_pwm=RC_NEUTRAL
            )
            time.sleep(0.1)

        print("Releasing RC override...")
        self.release_rc_override()

