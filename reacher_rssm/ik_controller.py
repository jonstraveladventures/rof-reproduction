# ===========================================================================
# Reacher RSSM - 2-link inverse kinematics + PD torque controller.
#
# Used by collect_dataset.py to generate the "good IK" and "noisy IK"
# trajectories. Pure Python / NumPy. No torch, no gymnasium imports.
#
# Conventions:
#   - Joints are revolute, planar (rotation around z).
#   - theta0 is the angle of link 1 relative to the world x-axis.
#   - theta1 is the angle of link 2 relative to link 1 (a.k.a. elbow angle).
#   - Fingertip position:
#       fx = l1*cos(theta0) + l2*cos(theta0 + theta1)
#       fy = l1*sin(theta0) + l2*sin(theta0 + theta1)
#   - Reachable workspace: |l1 - l2| <= r <= l1 + l2  with  r = ||target||
# ===========================================================================

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Forward kinematics (sanity-check helper, not used in the hot loop).
# ---------------------------------------------------------------------------
def forward_kinematics(theta0: float, theta1: float,
                       l1: float, l2: float) -> tuple[float, float]:
    """Return fingertip (x, y) given joint angles."""
    fx = l1 * math.cos(theta0) + l2 * math.cos(theta0 + theta1)
    fy = l1 * math.sin(theta0) + l2 * math.sin(theta0 + theta1)
    return fx, fy


# ---------------------------------------------------------------------------
# Inverse kinematics for a planar 2-link arm.
# ---------------------------------------------------------------------------
def inverse_kinematics_2link(target_x: float, target_y: float,
                             l1: float, l2: float,
                             elbow: str = "up") -> tuple[float, float]:
    """
    Solve theta0, theta1 such that the fingertip reaches (target_x, target_y).

    Args:
        target_x, target_y: target position in world frame.
        l1, l2: link lengths (Reacher-v5: l1=0.10, l2=0.11).
        elbow: 'up' or 'down'. Picks one of the two IK branches.

    Returns:
        (theta0, theta1) as floats in radians.

    If the target is outside the reachable disk, it is clamped onto the disk
    boundary (slightly inside the maximum reach to avoid singular Jacobians).
    """
    r2 = target_x * target_x + target_y * target_y
    r = math.sqrt(r2)

    # Reachability: clamp target onto the workspace if outside.
    r_max = (l1 + l2) * 0.999     # slight inset to avoid singularity
    r_min = abs(l1 - l2) * 1.001
    if r > r_max:
        scale = r_max / r
        target_x *= scale
        target_y *= scale
        r = r_max
        r2 = r * r
    elif r < r_min:
        # Very close to the base; pick a small radius and keep direction.
        if r > 1e-8:
            scale = r_min / r
            target_x *= scale
            target_y *= scale
        else:
            target_x, target_y = r_min, 0.0
        r = r_min
        r2 = r * r

    # Standard 2-link IK via the law of cosines.
    cos_theta1 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_theta1 = max(-1.0, min(1.0, cos_theta1))   # numerical safety
    theta1 = math.acos(cos_theta1)
    if elbow == "down":
        theta1 = -theta1

    # theta0 = atan2(y, x) - atan2(l2*sin(theta1), l1 + l2*cos(theta1))
    k1 = l1 + l2 * math.cos(theta1)
    k2 = l2 * math.sin(theta1)
    theta0 = math.atan2(target_y, target_x) - math.atan2(k2, k1)

    return theta0, theta1


# ---------------------------------------------------------------------------
# PD torque controller.
# ---------------------------------------------------------------------------
def pd_torque(theta_current: np.ndarray,
              theta_dot: np.ndarray,
              theta_target: np.ndarray,
              kp: float = 2.0,
              kd: float = 0.3,
              action_low: float = -1.0,
              action_high: float = 1.0) -> np.ndarray:
    """
    Compute clipped torques for both joints with a simple PD law.

    Args:
        theta_current: shape (2,), current joint angles.
        theta_dot: shape (2,), current joint angular velocities.
        theta_target: shape (2,), desired joint angles (e.g. from IK).
        kp, kd: scalar PD gains. Same gains for both joints.
        action_low, action_high: torque clip bounds.

    Returns:
        action: shape (2,) float32, clipped to [action_low, action_high].
    """
    error = _wrap_angle(theta_target - theta_current)
    torque = kp * error - kd * theta_dot
    return np.clip(torque, action_low, action_high).astype(np.float32)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]. Element-wise."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Convenience: extract joint angles + velocities from a Reacher-v5 obs.
# ---------------------------------------------------------------------------
# Reacher-v5 observation layout (10-D; v5 dropped the always-zero z-component
# that Reacher-v4 had at index 10):
#   [0]  cos(theta0)
#   [1]  cos(theta1)
#   [2]  sin(theta0)
#   [3]  sin(theta1)
#   [4]  target_x
#   [5]  target_y
#   [6]  theta_dot_0
#   [7]  theta_dot_1
#   [8]  fingertip_x - target_x
#   [9]  fingertip_y - target_y
def joint_angles_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return joint angles (theta0, theta1) recovered from cos/sin obs."""
    theta0 = math.atan2(float(obs[2]), float(obs[0]))
    theta1 = math.atan2(float(obs[3]), float(obs[1]))
    return np.array([theta0, theta1], dtype=np.float64)


def joint_velocities_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return joint velocities (theta_dot_0, theta_dot_1) from obs."""
    return np.array([float(obs[6]), float(obs[7])], dtype=np.float64)


def target_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return target (x, y) from obs."""
    return np.array([float(obs[4]), float(obs[5])], dtype=np.float64)


def fingertip_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return fingertip (x, y) by adding (fingertip - target) to target."""
    tgt = target_from_obs(obs)
    delta = np.array([float(obs[8]), float(obs[9])], dtype=np.float64)
    return tgt + delta


# ---------------------------------------------------------------------------
# Self-test (runs only when invoked directly, useful for sanity checks).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    L1, L2 = 0.10, 0.11
    print("Quick IK round-trip test (target -> IK -> FK -> target):")
    test_targets = [
        (0.10, 0.05),
        (-0.05, 0.15),
        (0.18, 0.0),
        (0.0, -0.10),
        (0.05, -0.18),
        (0.25, 0.0),    # outside workspace -> should be clamped
    ]
    for tx, ty in test_targets:
        t0, t1 = inverse_kinematics_2link(tx, ty, L1, L2, elbow="up")
        fx, fy = forward_kinematics(t0, t1, L1, L2)
        err = math.hypot(fx - tx, fy - ty)
        print(f"  target=({tx:+.3f},{ty:+.3f})  ->  theta=({t0:+.3f},{t1:+.3f})"
              f"  ->  fk=({fx:+.3f},{fy:+.3f})  err={err:.4f}"
              f"{'  (clamped)' if err > 1e-3 else ''}")

    print("\nPD torque sanity:")
    th = np.array([0.0, 0.0])
    thd = np.array([0.0, 0.0])
    th_t = np.array([0.5, -0.3])
    print(f"  pd_torque(theta=0, dot=0, target={th_t}) = {pd_torque(th, thd, th_t)}")
