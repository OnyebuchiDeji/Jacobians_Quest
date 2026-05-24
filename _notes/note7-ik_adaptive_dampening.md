####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 7: Inverse Kinematics - Dampening During Singularities Part 2

>    This contains the code to detect singularities and apply ADAPTIVE dampening to avoid joint angle explosions as can be seen in `jacob_ik.py` 

###	Me
Yes --- from previous notes chat response question at the end
To implement adaptive damping, you replace the fixed damping value with a variable factor. When the arm is far from a singularity, the damping drops to zero to maintain perfect tracking speed and accuracy. When the arm approaches a boundary, the damping smoothly increases to protect the system from mathematical explosion.
The standard industry method relies on the Manipulability Index ($\omega$), which is derived from the determinant.
## Script with Adaptive Damping
This script modifies your original tracking loop to dynamically calculate the damping factor ($\lambda$) during each iteration based on how close the arm is to a singularity.
```
import numpy as np
def adaptive_jacobian_ik(target_pos, l1, l2, q_current, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05):
    q = np.array(q_current, dtype=float)
    
    for i in range(max_iter):
        # 1. Forward Kinematics
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        current_pos = np.array([x, y])
        
        # 2. Error Calculation
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            print(f"Target reached in {i} iterations.")
            return q  
        
        # 3. Compute Jacobian Matrix
        J = np.array([
            [-l1*np.sin(q[0]) - l2*np.sin(q[0]+q[1]), -l2*np.sin(q[0]+q[1])],
            [ l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]),  l2*np.cos(q[0]+q[1])]
        ])
        
        # 4. Measure Manipulability (Using the analytical shortcut)
        # For a 2-link arm, the determinant directly measures manipulability
        det = l1 * l2 * np.sin(q[1])
        manipulability = np.abs(det)
        
        # 5. Adaptive Damping Law (Selectively apply damping)
        if manipulability < threshold:
            # Smoothly transition damping from 0 up to max_damping
            damping = max_damping * (1.0 - (manipulability / threshold))**2
        else:
            damping = 0.0 # Standard inverse is safe to use
            
        # 6. Damped Least Squares Update
        I = np.eye(2)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)
        
        dq = J_damped_inv @ error
        q += dq
        
    print("Max iterations reached without fully hitting tolerance.")
    return q
# Execution Matrix (Targeting a point exactly on the physical workspace limit)

target = np.array([2.0, 0.0])
initial_joints = [0.1, 0.01]
# Very close to fully extended straight arm
final_joints = adaptive_jacobian_ik(target, 1.0, 1.0, initial_joints)

print(f"Final joint angles: {np.degrees(final_joints)} degrees")
```


## How the Adaptive Logic Functions

* Far from Singularity ($\omega \ge \text{threshold}$): Damping becomes exactly 0.0. The calculation mathematically simplifies back to standard inversion, yielding the fastest path convergence.
* Near Singularity ($\omega < \text{threshold}$): Damping turns on smoothly using a parabolic scaling curve. It ensures a seamless transition without creating abrupt jerky movements in physical robot joints.
* At Peak Singularity ($\omega = 0$): Damping hits its hard ceiling (max_damping), capping joint velocities safely so the code cannot divide by zero.

Would you like to explore how to add joint limits to this solver to stop the arm from bending backward into unrealistic human positions?
