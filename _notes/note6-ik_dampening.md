####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 6: Inverse Kinematics - Dampening During Singularities Part 1

>    This contains the code to detect singularities and apply dampening to avoid joint angle explosions as can be seen in `jacob_ik.py` 
>   However, I use a better solution that implements adaptive dampening as seen in `note7-ik_adaptive_dampening.md`

### Chat Response

Joint velocities in the third code block are mathematically distinct from the final joint angles calculated in the first code block, though they are directly related.
Here is the breakdown of your questions, the unified code, and the comparison of the singularity checks.
## Joint Velocities vs. Joint Angles

* First Code Block (Joint Angles): Calculates the absolute final positions ($\mathbf{q}$) needed to reach a target. The step update dq represents a small change in angle ($\Delta\mathbf{q}$) per iteration step.
* Third Code Block (Joint Velocities): Calculates the rate of change of the joints ($\mathbf{\dot{q}}$) relative to time. It answers how fast the joints must move to achieve a specific tip velocity vector ($\mathbf{V}$), rather than a static destination point.

## Augmented Inverse Kinematics Script
This script integrates the Damped Least Squares (DLS) method from the third block into the iterative tracking loop of the first block. This prevents the joint updates from exploding near singularities.

import numpy as np
def jacobian_damped_ik(target_pos, l1, l2, q_current, max_iter=100, tol=1e-4, damping=0.1):
    q = np.array(q_current, dtype=float)
    
    for i in range(max_iter):
        # 1. Forward Kinematics: Current Tip Position
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        current_pos = np.array([x, y])
        
        # 2. Error Calculation (Acts as virtual displacement/velocity)
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            print(f"Target reached in {i} iterations.")
            return q  
        
        # 3. Compute Jacobian Matrix
        J = np.array([
            [-l1*np.sin(q[0]) - l2*np.sin(q[0]+q[1]), -l2*np.sin(q[0]+q[1])],
            [ l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]),  l2*np.cos(q[0]+q[1])]
        ])
        
        # 4. Singularity Check via Determinant
        det = np.linalg.det(J)
        if np.abs(det) < 1e-3:
            print(f"Warning: Close to singularity at iteration {i} (Det: {det:.6f}). Applying damping.")

        # 5. Damped Least Squares Update
        # Formula: dq = J.T @ inv(J @ J.T + lambda^2 * I) @ error
        I = np.eye(2)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)
        
        dq = J_damped_inv @ error
        q += dq
        
    return q
# Execution Matrixtarget = np.array([2.0, 0.0]) # Target lies exactly on the workspace boundary (Singularity)initial_joints = [0.1, 0.1]final_joints = jacobian_damped_ik(target, 1.0, 1.0, initial_joints)

print(f"Final joint angles: {np.degrees(final_joints)} degrees")

## Determinant Logic Comparison
The singularity check logic in the second block is mathematically identical to the logic in the third block, but it uses a different computational approach.
## Second Code Block (Analytical)

* Method: Uses the simplified geometric equation: $\text{det}(J) = l_1 \cdot l_2 \cdot \sin(\theta_2)$.
* Pros: Highly efficient. It bypasses matrix multiplication entirely. It isolates the exact joint ($\theta_2$) responsible for the singularity.

## Third Code Block (Numerical)

* Method: Uses np.linalg.det(J) to calculate the determinant from the compiled numerical matrix.
* Pros: Universal approach. This line of code remains unchanged even if you scale the robot arm up to 6 or 7 degrees of freedom.

If you want to optimize your code further, I can show you how to adaptively scale the damping factor so it only turns on exactly at the singularity, keeping the algorithm faster when far away. Would you like to see how to implement adaptive damping?


---
#####   OG Prompt
Following a conversation we had a while ago, I asked questions concerning inverse and forward kinematics and using Jacobian matrices. This prompt is specifically about using Jacobian matrices for inverse kinematics. You gave me these python source code: "```import numpy as np
def jacobian_ik(target_pos, l1, l2, q_current, max_iter=100, tol=1e-4):
    q = np.array(q_current, dtype=float)
    for i in range(max_iter):
        # 1. Forward Kinematics: Current Tip Position
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        current_pos = np.array([x, y])
        
        # 2. Error Calculation
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            return q  # Goal reached
        
        # 3. Compute Jacobian Matrix
        J = np.array([
            [-l1*np.sin(q[0]) - l2*np.sin(q[0]+q[1]), -l2*np.sin(q[0]+q[1])],
            [ l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]),  l2*np.cos(q[0]+q[1])]
        ])
        
        # 4. Joint Update: dq = J_inv * error
        # Use pinv (pseudoinverse) for better stability near singularities
        dq = np.linalg.pinv(J) @ error
        q += dq
        
    return q
# Usage
target = np.array([1.2, 0.8])
final_joints = jacobian_ik(target, 1.0, 1.0, [0.1, 0.1])
print(f"Target reached at joints: {np.degrees(final_joints)} degrees")
```
",
"```
def check_singularity(q, l1, l2):
    # Determinant of the 2-link Jacobian simplifies to: l1 * l2 * sin(theta2)
    # Singularity occurs when sin(theta2) = 0 (arm is fully stretched or folded)
    det = l1 * l2 * np.sin(q[1])
    
    if np.abs(det) < 1e-3:
        return True, det
    return False, det
# Example: Arm fully extended (theta2 = 0)is_singular, val = check_singularity([0.5, 0.0], 1.0, 1.0)
print(f"Is Singular? {is_singular} (Det: {val})")
```", and "
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
```"
The first demonstrates the algorithm for solving for the final joint angles for links/arms to reach atarget point for inverse kinematics. The next shows a function for calculating "singularity", precisely to detect it, and the third is an algorithm that also includes a check for singularity and regulating it with dampening. Given all these, first clarify that what is referred to as "joint velocities" in the third code block is the same as the joint angles being calculated in the first code block. Second, using the first code block as a basis, augment its logic by adding the singularity check and dampening logic demonstrated in the third code block and write the python script that shows this. Finally, consider the singularity check logic of the second code block; is it different from the singularity check logic used in the third code block? Please explain.