####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 9: Inverse Kinematics - 3-Link Arm

>	Implementation of inverse kinematics formula for a 3-link/4-joint arm

>   The 3-link arm has many joints and thus much more degrees of freedom
>   Due to this, the arm's shape can change very much.

>	For future. Not yet implemented

###	Me
Yes --- from previous notes chat response question at the end


###	Chat
Extending this to a 3-link arm changes the problem from determined (2 inputs, 2 outputs) to redundant (3 inputs, 2 outputs) [1].
Because the arm has an extra joint, it has infinite ways to reach the same 2D coordinate [1]. The Damped Least Squares (DLS) formula handles this automatically. It picks the path that requires the least overall joint movement [1].
## 3-Link Script with Damping and Limits
This script expands the matrices to $2 \times 3$ and adds a third joint limit constraint.

```

import numpy as np
def limited_3link_ik(target_pos, l1, l2, l3, q_current, q_min, q_max, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05):
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    
    for i in range(max_iter):
        # 1. Forward Kinematics for a 3-Link Arm
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
        
        # 4. Measure Manipulability for Redundant Robot
        # For non-square matrices, use det(J @ J.T) instead of det(J)
        manipulability = np.sqrt(np.abs(np.linalg.det(J @ J.T)))
        
        # 5. Adaptive Damping Law
        if manipulability < threshold:
            damping = max_damping * (1.0 - (manipulability / threshold))**2
        else:
            damping = 0.0
            
        # 6. Damped Least Squares Update (Redundant Formulation)
        # Formula: dq = J.T @ inv(J @ J.T + lambda^2 * I) @ error
        I = np.eye(2) # Identity matrix matches the workspace dimensions (X, Y)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)
        dq = J_damped_inv @ error
        
        # 7. Apply Joint Limits (Clamping)
        q_next = q + dq
        q = np.clip(q_next, q_min, q_max)
        
    print("Max iterations reached. Returned best safe 3-link configuration.")
    return q
# --- Execution Setup ---link1, link2, link3 = 1.0, 1.0, 0.5
# Define limits for all three jointsmin_limits = [np.radians(-90), np.radians(-90), np.radians(-90)]max_limits = [np.radians(90),  np.radians(90),  np.radians(90)]
# Target pointtarget = np.array([1.5, 0.5]) initial_joints = [0.1, 0.1, 0.1]
final_joints = limited_3link_ik(target, link1, link2, link3, initial_joints, min_limits, max_limits)

print(f"Final 3-joint angles: {np.degrees(final_joints)} degrees")
```
## Key Changes for Redundancy

* Non-Square Jacobian ($2 \times 3$): The matrix has 2 rows (X, Y tracking) and 3 columns (one for each joint).
* New Manipulability Formula: You cannot take the determinant of a non-square matrix directly. Instead, you multiply it by its transpose ($\det(J \cdot J^T)$) to compress it into a square $2 \times 2$ shape first.
* The Identity Matrix ($I$): The size of the identity matrix matches the task dimension ($2 \times 2$ for X and Y), not the joint count.

Because this arm is redundant, we can now use the extra movement freedom to achieve secondary goals. Would you like to see how to program the arm to actively avoid its own joint limits while tracking the target?

