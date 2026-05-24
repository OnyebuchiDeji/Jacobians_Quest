####  Date: 12-03-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 5

####  Me
How then can the singularities be used to regulate the 2D arm's kinematics --- is there a difference when regulating singulariting the singularity for forward kinematics in contrast with inverse kinematics? Explain and demonstrate this in a Python script utilizing only numpy as well.

####   Response from Google AI

In kinematics, singularities affect forward and inverse operations in opposite ways.
The Contrast in Regulation

   1. Forward Kinematics (Joints $\to$ Tip):
   * The Problem: Singularities aren't "dangerous" here. The math ($V = J\dot{q}$) always works.
      * The Observation: At a singularity, you lose a dimension of motion. No matter how fast you spin your joints, the tip velocity in a certain direction will be zero.
      * Regulation: Used to limit expectations. You use the Jacobian to realize the robot cannot move in a specific direction (e.g., if the arm is straight, it can't move further "out").
   2. Inverse Kinematics (Tip $\to$ Joints):
   * The Problem: This is where the danger lies. The math ($\dot{q} = J^{-1}V$) "explodes."
      * The Observation: As the determinant approaches zero, $J^{-1}$ approaches infinity. Small desired tip movements command infinite joint speeds.
      * Regulation: Used for stabilization. We use Damped Least Squares (DLS) to "brake" the joints. We sacrifice tracking accuracy to ensure the robot doesn't break itself.
   
------------------------------
Python Demonstration: Regulation via Damping
This script shows how to "regulate" a singularity during Inverse Kinematics using a damping factor ($\lambda$).

```
import numpy as np
def regulated_ik_step(target_vel, q, l1, l2, damping=0.1):
    """
    Calculates joint velocities given a desired tip velocity.
    Uses Damped Least Squares to regulate singularities.
    """
    q1, q2 = q
    
    # 1. Compute Jacobian
    J = np.array([
        [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
        [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2)]
    ])
    
    # 2. Check Determinant (Forward Kinematics Perspective)
    det = np.linalg.det(J)
    
    # 3. Regulate Inverse Kinematics
    # Standard: dq = inv(J) @ V (Explodes near singularity)
    # Damped:   dq = J.T @ inv(J@J.T + lambda^2 * I) @ V
    
    # Identity matrix
    I = np.eye(2)
    
    # Damped Least Squares formula (Levenberg-Marquardt style)
    # This prevents the inversion from reaching infinity
    J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)
    joint_vels = J_damped_inv @ target_vel
    
    return joint_vels, det
```
# --- Scenario: Arm nearly fully extended (Near Singularity) ---
l1, l2 = 1.0, 1.0q_near_sing = [0.1, 0.001] # Almost a straight linedesired_tip_vel = np.array([0.5, 0.0]) # Try to push "further out"
# Compare Undamped vs Damped
dq_unsafe, _ = regulated_ik_step(desired_tip_vel, q_near_sing, l1, l2, damping=0.0)
dq_safe, det = regulated_ik_step(desired_tip_vel, q_near_sing, l1, l2, damping=0.1)

print(f"Jacobian Determinant: {det:.6f} (Very Close to 0)")
print(f"UNREGULATED Joint Velocities: {dq_unsafe} <--- DANGEROUSLY HIGH")
print(f"REGULATED Joint Velocities:   {dq_safe}   <--- CONTROLLED")

Key Takeaways

* The Damping Factor ($\lambda$): When the robot is in a "healthy" configuration, the damping has almost no effect. When the determinant drops, the damping term dominates the denominator, preventing the result from skyrocketing.
* The Trade-off: With regulation, the robot won't move exactly at 0.5 m/s as requested, but it will move as fast as safely possible without the motors screaming.

Would you like to see how to visualize these "manipulability ellipses" to see which directions are losing velocity in real-time?

