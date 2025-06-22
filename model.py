import numpy as np

def dh_transform(a, alpha, d, theta):
    """Denavit-Hartenberg transformation matrix"""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),                d],
        [0,              0,                           0,                            1]
    ])

# Standard DH parameters for Franka Panda (7-DOF)
DH_PARAMS = [
    (0,       -np.pi/2, 0.333),     # Joint 1
    (0,        np.pi/2, 0),         # Joint 2
    (0,        np.pi/2, 0.316),     # Joint 3
    (0.0825,   np.pi/2, 0),         # Joint 4
    (-0.0825, -np.pi/2, 0.384),     # Joint 5
    (0,        np.pi/2, 0),         # Joint 6
    (0.088,    np.pi/2, 0),         # Joint 7
]

def forward_kinematics(q):
    """Compute Franka Panda end-effector pose from joint angles q (7,)"""
    T = np.eye(4)
    for i in range(7):
        a, alpha, d = DH_PARAMS[i]
        T = T @ dh_transform(a, alpha, d, q[i])
    return T  # 4x4 homogeneous transformation

# Example usage:
q = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
T_ee = forward_kinematics(q)
print(T_ee)
