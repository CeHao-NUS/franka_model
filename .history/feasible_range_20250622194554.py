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


# Number of random samples
N = 50000  # Increase for finer visualization

positions = []
for _ in range(N):
    # Random joint config within limits
    q = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
    T_ee = forward_kinematics(q)
    pos = T_ee[:3, 3]
    positions.append(pos)
positions = np.array(positions)

# Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=positions[:,2], cmap='viridis', s=1, alpha=0.4)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Estimated Franka Panda Reachable Workspace')
ax.set_box_aspect([1,1,1])
# plt.show()
plt.savefig("feasible_region.png", dpi=300)

from scipy.spatial import ConvexHull

# Compute convex hull
hull = ConvexHull(positions)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot points (optional, for context)
ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='lightgrey', s=0.1, alpha=0.1)

# Plot hull triangles
for simplex in hull.simplices:
    tri = positions[simplex]
    ax.plot_trisurf(tri[:,0], tri[:,1], tri[:,2], color='cyan', alpha=0.3, linewidth=0)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Franka Panda Workspace Convex Hull')
ax.set_box_aspect([1,1,1])
plt.savefig('workspace_hull.png', dpi=300)
