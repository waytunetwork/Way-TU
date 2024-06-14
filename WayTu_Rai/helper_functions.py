import numpy as np
import math

def rotate_object (yaw):
    # A rotation translation from degree (yaw) to quaternion
    qw = np.cos(yaw / 2.0)
    qx = 0.0
    qy = 0.0
    qz = np.sin(yaw / 2.0)

    return np.array([qw, qx, qy, qz])


def quaternion_rotation_on_z_axis(quaternion, angle_degrees):
    # Rotate a quaternion around the Z-axis by a given angle.

    angle_radians = math.radians(angle_degrees)
    half_angle = angle_radians / 2
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)

    # Rotation quaternion for Z-axis
    rot_quaternion = (cos_half_angle, 0, 0, sin_half_angle)

    # Quaternion multiplication (rot_quaternion * quaternion)
    w1, x1, y1, z1 = rot_quaternion
    w2, x2, y2, z2 = quaternion
    rotated_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    rotated_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    rotated_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    rotated_z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (rotated_w, rotated_x, rotated_y, rotated_z)


def quaternion_rotation(quaternion, angle_degrees, axis):
    # Convert angle to radians and calculate half angle
    angle_radians = math.radians(angle_degrees)
    half_angle = angle_radians / 2
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)

    # Normalize the axis
    axis_length = math.sqrt(sum([x**2 for x in axis]))
    normalized_axis = tuple(x/axis_length for x in axis)

    # Rotation quaternion for the arbitrary axis
    rot_quaternion = (
        cos_half_angle,
        normalized_axis[0] * sin_half_angle,
        normalized_axis[1] * sin_half_angle,
        normalized_axis[2] * sin_half_angle,
    )

    # Quaternion multiplication (rot_quaternion * quaternion)
    w1, x1, y1, z1 = rot_quaternion
    w2, x2, y2, z2 = quaternion
    rotated_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    rotated_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    rotated_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    rotated_z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (rotated_w, rotated_x, rotated_y, rotated_z)