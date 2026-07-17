#!/usr/bin/env python3
import math

from config_blueboat import (LAT0_DEG, LON0_DEG, PSI0_RAD, X0_LOCAL, Y0_LOCAL,)

# Earth radius in meters
R_EARTH = 6378137.0


def gps_to_xy(lat_deg, lon_deg, lat0_deg=LAT0_DEG, lon0_deg=LON0_DEG, psi0=PSI0_RAD, x0_local=X0_LOCAL, y0_local=Y0_LOCAL):

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    x_east = R_EARTH * math.cos(lat0) * (lon - lon0)
    y_north = R_EARTH * (lat - lat0)

    x_relative = math.cos(psi0) * x_east - math.sin(psi0) * y_north
    y_relative = math.sin(psi0) * x_east + math.cos(psi0) * y_north

    x = x0_local + x_relative
    y = y0_local + y_relative

    return x, y


def yaw_attitude_rad_to_theta_local(yaw_rad, psi0=PSI0_RAD):

    v_east = math.sin(yaw_rad)
    v_north = math.cos(yaw_rad)

    v_x = math.cos(psi0) * v_east - math.sin(psi0) * v_north
    v_y = math.sin(psi0) * v_east + math.cos(psi0) * v_north

    theta = math.atan2(v_y, v_x)

    return theta


def theta_local_to_heading_deg(theta, psi0=PSI0_RAD):

    v_x = math.cos(theta)
    v_y = math.sin(theta)

    v_east = math.cos(psi0) * v_x + math.sin(psi0) * v_y
    v_north = -math.sin(psi0) * v_x + math.cos(psi0) * v_y

    heading = math.atan2(v_east, v_north)
    heading_deg = math.degrees(heading) % 360.0

    return heading_deg

