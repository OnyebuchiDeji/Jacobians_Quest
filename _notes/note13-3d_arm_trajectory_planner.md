###  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 13: 3D 3-Link Arm Trajectory Planner

>	The 3-link arm has many joints and thus much more degrees of freedom
>	Due to this, the arm's shape can change very much.
>	The below explains how to make the 3D arm trace a sequence of coordinates (a trajectory) in space to create perform an action e.g. draw a pattern.
>	For future. Not yet implemented

###	Me
Yes --- from previous notes chat response question at the end



###	Chat

To move a 6-axis robot arm along a smooth geometric path (like painting a straight line or sealing a weld), we cannot just calculate a single target point. We must divide the path into a sequence of closely spaced target points called waypoints.
This is called Trajectory Tracking. By feeding each waypoint sequentially into our 3D solver and using the final joint angles of the previous step as the starting guess for the next step, the arm moves continuously and smoothly along the path.
## 3D Trajectory Tracking Script
This script generates a 3D linear trajectory (a straight line in 3D space) and commands the robot arm to track it step-by-step.
```
import numpy as np
def forward_kinematics_3d(q, lengths):
    l1, l2, l3, l4 = lengths
    q1, q2, q3, q4, q5, q6 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    
    z = l1 
    x = l2 * c2 * c1
    y = l2 * c2 * s1
    z += l2 * s2
    
    x += l3 * np.cos(q2 + q3) * c1
    y += l3 * np.cos(q2 + q3) * s1
    z += l3 * np.sin(q2 + q3)
    
    x += l4 * np.cos(q2 + q3 + q4) * c1
    y += l4 * np.cos(q2 + q3 + q4) * s1
    z += l4 * np.sin(q2 + q3 + q4)
    
    roll = q5
    pitch = q2 + q3 + q4
    yaw = q1 + q6
    
    return np.array([x, y, z, roll, pitch, yaw])
def numerical_jacobian_3d(q, lengths, h=1e-5):
    J = np.zeros((6, 6))
    f_current = forward_kinematics_3d(q, lengths)
    for i in range(6):
        q_perturbed = np.array(q, dtype=float)
        q_perturbed[i] += h
        f_perturbed = forward_kinematics_3d(q_perturbed, lengths)
        J[:, i] = (f_perturbed - f_current) / h
    return J
def spatial_3d_ik_step(target_pose, lengths, q_current, q_min, q_max, max_iter=50, tol=1e-3, damping=0.1):
    """
    Solves IK for a single waypoint. Uses q_current as a highly accurate warm-start guess.
    """
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    
    for i in range(max_iter):
        current_pose = forward_kinematics_3d(q, lengths)
        error = target_pose - current_pose
        error[3:] = (error[3:] + np.pi) % (2 * np.pi) - np.pi
        
        if np.linalg.norm(error) < tol:
            return q, True
            
        J = numerical_jacobian_3d(q, lengths)
        manipulability = np.abs(np.linalg.det(J))
        
        I = np.eye(6)
        lam = damping if manipulability < 0.01 else 0.0
        
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + lam**2 * I)
        dq = J_damped_inv @ error
        q = np.clip(q + dq, q_min, q_max)
        
    return q, False
# --- Execution Setup ---arm_links = [0.5, 1.0, 1.0, 0.3] min_limits = [np.radians(-180)] * 6max_limits = [np.radians(180)] * 6
# 1. Generate the 3D Trajectory (A straight line from Start Pose to End Pose)start_pose = np.array([1.0, 0.5, 1.0, 0.0, 0.5, 0.0])end_pose   = np.array([1.4, -0.2, 1.2, 0.0, 0.5, 0.0]) # Hand orientation stays fixed
num_steps = 10# Linear interpolation (LERP) between start and end pointstrajectory = [start_pose + (end_pose - start_pose) * (t / (num_steps - 1)) for t in range(num_steps)]
# 2. Track the Trajectory# Seed the very first step with a default neutral joint guesscurrent_joint_state = np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0])joint_history = []

print("Starting 3D Trajectory Tracking...")for step, waypoint in enumerate(trajectory):
    # Pass the previous step's joints into the next step as the initial guess
    current_joint_state, success = spatial_3d_ik_step(waypoint, arm_links, current_joint_state, min_limits, max_limits)
    
    if success:
        joint_history.append(current_joint_state)
        # Verify the actual position reached via Forward Kinematics
        reached_pos = forward_kinematics_3d(current_joint_state, arm_links)[:3]
        print(f"Step {step}: Success -> Tip at X:{reached_pos[0]:.2f}, Y:{reached_pos[1]:.2f}, Z:{reached_pos[2]:.2f}")
    else:
        print(f"Step {step}: Failed to converge closely enough to waypoint.")

print(f"\nTracking complete. Generated smooth profiles for {len(joint_history)} physical configuration sets.")
```
## Key Concepts of Trajectory Tracking

* Warm Starting (Temporal Continuity): The secret to smooth movement is setting q_current to the previous loop's answer. Because the waypoints are physically close together, the solver usually finds the next answer in just 2 to 5 iterations.
* Preventing Joint Flips: Because inverse kinematics has multiple geometric solutions, warm starting ensures the arm chooses the solution closest to its current shape, preventing the joints from violently flipping or rotating $180^\circ$ mid-path.
* Constant Orientation: By keeping the orientation parameters (Roll, Pitch, Yaw) identical across all waypoints, the robot translates its arm through space while holding its tool completely level.

Would you like to see how to implement Velocity Limiting to guarantee that the joints do not move faster than the actual physical motors can spin?

