import numpy as np
import matplotlib.pyplot as plt

# Joint limits for 7-DOF Franka Panda (in radians)
joint_limits = [
    (-2.8973,  2.8973),    # Joint 1
    (-1.7628,  1.7628),    # Joint 2
    (-2.8973,  2.8973),    # Joint 3
    (-3.0718, -0.0698),    # Joint 4
    (-2.8973,  2.8973),    # Joint 5
    (-0.0175,  3.7525),    # Joint 6
    (-2.8973,  2.8973),    # Joint 7
]

def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),                d],
        [0,              0,                           0,                            1]
    ])

DH_PARAMS = [
    (0,       -np.pi/2, 0.333),
    (0,        np.pi/2, 0),
    (0,        np.pi/2, 0.316),
    (0.0825,   np.pi/2, 0),
    (-0.0825, -np.pi/2, 0.384),
    (0,        np.pi/2, 0),
    (0.088,    np.pi/2, 0),
]

def forward_kinematics(q):
    T = np.eye(4)
    for i in range(7):
        a, alpha, d = DH_PARAMS[i]
        T = T @ dh_transform(a, alpha, d, q[i])
    return T  # 4x4 homogeneous transformation


