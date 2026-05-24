####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 11: 3-Link Arm Obstacle Avoidance

>	The 3-link arm has many joints and thus much more degrees of freedom
>	Due to this, the arm's shape can change very much.
>	The below explains how to make the arm's shape change to avoid obstacles along the way
>	For future. Not yet implemented

###	Me
Yes --- from previous notes chat response question at the end

###	Chat

To handle obstacle avoidance using the null space, we use a technique called the Potential Field Method.
We treat the obstacle as a magnetic source of repulsion. We calculate a "push" vector away from the obstacle for each joint or link, and then project that vector into the arm's null space. This allows the internal joints of the arm to dynamically bend and dodge an object while keeping the tip perfectly locked onto the target or moving smoothly toward it.
## 3-Link IK Script with Obstacle Avoidance
This script simulates a circular obstacle in the 2D workspace. It calculates the distance from the middle joint (the elbow) to the obstacle and pushes it away if it gets too close.
```
import numpy as np
def obstacle_avoidance_ik(target_pos, l1, l2, l3, q_current, q_min, q_max, obstacle_pos, obs_radius, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05, k_avoid=1.0):
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    
    # Safe distance from obstacle boundary (buffer zone)
    influence_dist = 0.4 
    
    for i in range(max_iter):
        # 1. Forward Kinematics (Positions of the joints to track them)
        q1, q2, q3 = q[0], q[1], q[2]
        
        # Position of Joint 2 (Elbow 1)
        p1 = np.array([l1*np.cos(q1), l1*np.sin(q1)])
        # Position of Joint 3 (Elbow 2)
        p2 = p1 + np.array([l2*np.cos(q1+q2), l2*np.sin(q1+q2)])
        # Position of End Effector (Tip)
        p3 = p2 + np.array([l3*np.cos(q1+q2+q3), l3*np.sin(q1+q2+q3)])
        
        # 2. Error Calculation
        error = target_pos - p3
        if np.linalg.norm(error) < tol:
            print(f"Target reached in {i} iterations.")
            return q  
        
        # 3. Compute 2x3 Jacobian Matrix for the Tip
        J = np.array([
            [-l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3), -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3), -l3*np.sin(q1+q2+q3)],
            [ l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),  l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),  l3*np.cos(q1+q2+q3)]
        ])
        
        # 4. Measure Manipulability & Damping
        manipulability = np.sqrt(np.abs(np.linalg.det(J @ J.T)))
        damping = max_damping * (1.0 - (manipulability / threshold))**2 if manipulability < threshold else 0.0
            
        # 5. Primary Task: Compute Damped Pseudoinverse
        I_task = np.eye(2)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I_task)
        dq_primary = J_damped_inv @ error
        
        # 6. Secondary Task: Repulsive Field from Obstacle
        # We will monitor Joint 3 (p2) to make sure it doesn't hit the obstacle
        dist_to_obs = np.linalg.norm(p2 - obstacle_pos) - obs_radius
        
        dq_secondary_goal = np.zeros(3)
        if dist_to_obs < influence_dist:
            # Vector pointing directly away from the obstacle center
            repulsion_dir = (p2 - obstacle_pos) / np.linalg.norm(p2 - obstacle_pos)
            
            # Magnitude increases as the arm gets closer to the obstacle
            repulsion_mag = k_avoid * (1.0 / max(dist_to_obs, 0.001) - 1.0 / influence_dist)
            virtual_force = repulsion_dir * repulsion_mag
            
            # Jacobian for Joint 3 (p2) to translate the Cartesian force into joint space
            J_p2 = np.array([
                [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2), 0],
                [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2), 0]
            ])
            # Project the Cartesian repulsion force into joint speeds
            dq_secondary_goal = J_p2.T @ virtual_force
            
        # 7. Null Space Projection Matrix
        I_joint = np.eye(3)
        P = I_joint - (J_damped_inv @ J)
        dq_null_space = P @ dq_secondary_goal
        
        # 8. Combine and Apply Clamped Step
        dq = dq_primary + dq_null_space
        q = np.clip(q + dq, q_min, q_max)
        
    print("Max iterations reached. Returned safest possible bypass route.")
    return q
# --- Execution Setup ---link1, link2, link3 = 1.0, 1.0, 0.5min_limits = [np.radians(-180)] * 3max_limits = [np.radians(180)] * 3
# Define a circular obstacle blocking a direct line-of-sight pathobstacle_center = np.array([1.0, 0.2])obstacle_radius = 0.25
# Target requires the arm to reach past or around the obstacletarget = np.array([1.8, 0.0]) initial_joints = [0.4, -0.2, -0.2] # Initial guess shape
final_joints = obstacle_avoidance_ik(target, link1, link2, link3, initial_joints, min_limits, max_limits, obstacle_center, obstacle_radius)
print(f"Final 3-joint angles: {np.degrees(final_joints)} degrees")
```

## How the Obstacle Avoidance Matrix Operates

* The Buffer Zone (influence_dist): If the arm is far from the object, the secondary goal defaults to 0.0. It behaves identically to standard tracking. It only creates a physical reaction when entering the field.
* The Sub-Jacobian Matrix (J_p2): Because the obstacle threatens the middle of the arm and not the tip, we build a mini-Jacobian specifically tracking how the joints affect that specific elbow's position (p2). Joint 3 has a 0 value in this sub-Jacobian because moving the hand tip cannot alter the location of the elbow behind it.
* The Null Space Filter: The repulsion code generates a joint speed command (dq_secondary_goal). If left raw, this movement would knock the tip completely off the target. Multiplying it by the projection filter P ensures that the arm only uses its extra "slack" configurations to swerve around the object.

Would you like to step up from 2D coordinates into a fully functional 3D space solver (X, Y, Z) with roll, pitch, and yaw tracking?

