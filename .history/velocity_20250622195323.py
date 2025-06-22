import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# Number of samples
N = 5000

# Joint limits for 7-DOF Franka Panda (in radians)
joint_limits = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
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

# Joint velocity limits (from Franka docs, in rad/s)
joint_vel_limits = np.array([
    2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100,
])

def jacobian_linear(q):
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    zs = [T[:3, 2].copy()]
    for i in range(7):
        a, alpha, d = DH_PARAMS[i]
        T = T @ dh_transform(a, alpha, d, q[i])
        origins.append(T[:3, 3].copy())
        zs.append(T[:3, 2].copy())
    o_n = origins[-1]
    Jv = np.zeros((3, 7))
    for i in range(7):
        Jv[:, i] = np.cross(zs[i], o_n - origins[i])
    return Jv

# Sampling loop
positions = []
max_lin_vels = []
qdot_max = np.linalg.norm(joint_vel_limits)
for _ in range(N):
    q = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
    T_ee = forward_kinematics(q)
    pos = T_ee[:3, 3]
    J = jacobian_linear(q)
    sigma_max = svd(J, compute_uv=False)[0]
    v_max = sigma_max * qdot_max
    positions.append(pos)
    max_lin_vels.append(v_max)

positions = np.array(positions)
max_lin_vels = np.array(max_lin_vels)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=max_lin_vels, cmap='inferno', s=1, alpha=0.6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(sc, ax=ax, label='Max Linear Velocity (m/s)')
ax.set_title('Max End-Effector Linear Velocity at Each Position')
plt.savefig('ee_max_velocity.png', dpi=300)
