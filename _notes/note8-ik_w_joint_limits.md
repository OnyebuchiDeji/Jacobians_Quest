####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 8: Inverse Kinematics - Dampening During Singularities and Joint Limits

>    This contains the code to detect singularities and apply ADAPTIVE dampening to avoid joint angle explosions and also includes joint angle limits as can be seen in `jacob_ik_2.py` 

###	Me
Yes --- from previous notes chat response question at the end


To enforce physical boundaries and stop the arm from bending backward or exceeding its mechanical range, you can implement Joint Limit Clamping.
The most robust way to do this in an iterative Jacobian solver is to calculate your damped joint step (dq), apply it to a temporary variable, clamp that variable to your hard hardware limits, and then calculate the actual allowed change. This prevents the error tracking from getting desynchronised.
## Kinematics Script with Adaptive Damping and Joint Limits
This script adds hard minimum and maximum constraints for both joints. For example, it restricts Joint 2 ($\theta_2$) to prevent it from flipping backward.

```
import numpy as np
def limited_adaptive_ik(target_pos, l1, l2, q_current, q_min, q_max, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05):
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    
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
        
        # 4. Measure Manipulability
        det = l1 * l2 * np.sin(q[1])
        manipulability = np.abs(det)
        
        # 5. Adaptive Damping Law
        if manipulability < threshold:
            damping = max_damping * (1.0 - (manipulability / threshold))**2
        else:
            damping = 0.0
            
        # 6. Damped Least Squares Update
        I = np.eye(2)
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)
        dq = J_damped_inv @ error
        
        # 7. Apply Joint Limits (Clamping)
        q_next = q + dq
        q_clamped = np.clip(q_next, q_min, q_max)
        
        # Update q with the safely restricted joint positions
        q = q_clamped
        
    print("Max iterations reached. Returned best safe configuration.")
    return q

# --- Execution Setup ---link1 = 1.0link2 = 1.0
# Define hard boundaries (in radians)
# Joint 1: Can rotate 180 degrees (-90 to +90)
# Joint 2: Restricted to positive angles only (0 to 110 degrees) to avoid bending backwards
min_limits = [np.radians(-90), np.radians(0)]
max_limits = [np.radians(90), np.radians(110)]
# Target point that would normally require Joint 2 to bend backwards to reach comfortably
target = np.array([0.5, -0.5])
initial_joints = [0.1, 0.2]
final_joints = limited_adaptive_ik(target, link1, link2, initial_joints, min_limits, max_limits)

print(f"Final joint angles: {np.degrees(final_joints)} degrees")
```

## Critical Rules for Joint Clamping

* Clamp the Total Position: Always clamp the absolute position (q + dq), not just the step size (dq). Clamping dq directly can still let the joints drift past boundaries over multiple steps.
* Avoid Matrix Locking: If a joint hits a limit, it stops updating. In advanced systems, you can zero out the corresponding column in the Jacobian matrix to tell the solver that the specific joint cannot contribute any more movement.

Would you like to see how to extend this logic to a 3-link arm, where we have an extra joint to help navigate around these limits?

