import numpy as np
from numpy.linalg import svd

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
