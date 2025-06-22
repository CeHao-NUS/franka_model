import numpy as np
from numpy.linalg import svd
from feasible_range import *

# Joint velocity limits (from Franka docs, in rad/s)
joint_vel_limits = np.array([
    2.1750,  # Joint 1
    2.1750,  # Joint 2
    2.1750,  # Joint 3
    2.1750,  # Joint 4
    2.6100,  # Joint 5
    2.6100,  # Joint 6
    2.6100,  # Joint 7
])

# -- Sampled positions and their associated joint configs
positions = []
max_lin_vels = []

for _ in range(N):
    q = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
    T_ee = forward_kinematics(q)
    pos = T_ee[:3, 3]
    # ---- Insert your Jacobian function here ----
    J = jacobian_linear(q)  # Should return 3x7 matrix
    # SVD for max singular value
    sigma_max = svd(J, compute_uv=False)[0]
    # Euclidean norm of velocity limits
    qdot_max = np.linalg.norm(joint_vel_limits)
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
