####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 10: 3-Link Arm Joint Limit Avoidance


>	The 3-link arm has many joints and thus much more degrees of freedom
>	Due to this, the arm's shape can change very much.
>	The below shows the implementation that makes the joints of the arm avoid their limits so that the solution for the tip is not disturbed

>	For future. Not yet implemented

###	Me
Yes --- from previous notes chat response question at the end

###	Chat

To make a redundant arm actively avoid its joint limits while tracking a target, you use a technique called Null Space Projection.
Because a 3-link arm has an extra degree of freedom, there are internal joint movements that change the shape of the arm without moving the tip at all. This is called the "null space" of the Jacobian. By pushing a secondary optimization goal into this null space, the arm will actively push its joints away from their limits as it moves.
## 3-Link IK Script with Null Space Limit Avoidance
This script calculates a secondary "push" vector that guides joints toward their middle positions and projects it so it never interferes with the primary hand tracking.
```
import numpy as np
def null_space_limit_avoidance_ik(target_pos, l1, l2, l3, q_current, q_min, q_max, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05, k_avoid=0.5):
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    q_center = (q_min + q_max) / 2.0  # Safe middle position for each joint
    
    for i in range(max_iter):
        # 1. Forward Kinematics
        q1, q2, q3 = q[0], q[1], q[2]
        x = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
        y = l1*np.sin(q1) + l2*np.sin(q1+q2) + l3*np.sin(q1+q2+q3)
        current_pos = np.array([x, y])
        
        # 2. Error Calculation
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            print(f"Target reached in {i} iterations.")
            return q  
        
        # 3. Compute 2x3 Jacobian Matrix
        J = np.array([
            [-l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3), -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3), -l3*np.sin(q1+q2+q3)],
            [ l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),  l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),  l3*np.cos(q1+q2+q3)]
        ])
        
        # 4. Measure Manipulability
        manipulability = np.sqrt(np.abs(np.linalg.det(J @ J.T)))
        
        # 5. Adaptive Damping Law
        damping = max_damping * (1.0 - (manipulability / threshold))**2 if manipulability < threshold else 0.0
            
        # 6. Primary Task: Compute Damped Pseudoinverse
        I_task = np.eye(2)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I_task)
        dq_primary = J_damped_inv @ error
        
        # 7. Secondary Task: Null Space Limit Avoidance
        # Generate a virtual velocity pushing joints back toward their center positions
        dq_secondary_goal = k_avoid * (q_center - q)
        
        # Identity matrix for the joint space (3x3)
        I_joint = np.eye(3)
        
        # Null Space Projection Matrix: P = I - (J_inv @ J)
        # This filters out any joint speeds that would alter the tip's position
        P = I_joint - (J_damped_inv @ J)
        dq_null_space = P @ dq_secondary_goal
        
        # 8. Combine and Apply Clamped Step
        dq = dq_primary + dq_null_space
        q = np.clip(q + dq, q_min, q_max)
        
    print("Max iterations reached. Returned best safe configuration.")
    return q
# --- Execution Setup ---link1, link2, link3 = 1.0, 1.0, 0.5
# Joint 3 is heavily restricted, forcing the null space to keep it centeredmin_limits = [np.radians(-90), np.radians(-90), np.radians(-15)]max_limits = [np.radians(90),  np.radians(90),  np.radians(15)]
target = np.array([1.2, 0.6]) initial_joints = [0.1, 0.1, 0.0]
final_joints = null_space_limit_avoidance_ik(target, link1, link2, link3, initial_joints, min_limits, max_limits)
print(f"Final 3-joint angles: {np.degrees(final_joints)} degrees")
```
## How Null Space Projection Operates

* The Projection Matrix ($P$): The math line P = I - (J_inv @ J) creates a directional filter. Any joint movement vector multiplied by $P$ will be stripped of its ability to move the end effector.
* The Secondary Goal (dq_secondary_goal): This acts like a spring pulling the joints toward the middle of their safe operating ranges (q_center).
* The Synthesis: The final step combines dq_primary (which moves the hand toward the target coordinate) and dq_null_space (which untangles the arm's shape internally without pulling the hand off-target).

Would you like to see how to adapt this solver to handle obstacle avoidance, allowing the arm to bend around fixed blocks in its workspace?

