####  Date: 24-05-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 12: 3D Arm Kinematics

>	The 3-link arm has many joints and thus much more degrees of freedom
>	Due to this, the arm's shape can change very much.
>	The below explains and shows solutions for the kinematics of a 3D arm
>	For future. Not yet implemented

###	Me
Yes --- from previous notes chat response question at the end


###	Chat

Moving up to 3D space changes our robot from a planar arm into a spatial manipulator (like an industrial 6-axis arm).
To fully control an object in 3D space, we must manage 6 degrees of freedom (DoF): 3 for translation ($X, Y, Z$) and 3 for orientation (Roll, Pitch, Yaw). This means our Jacobian matrix scales up to a size of $6 \times N$, where $N$ is the number of joints.
For this script, we will design a 6-DoF Articulated Arm.
## 3D Spatial Kinematics Script
Because calculating a 6-axis spatial Jacobian analytically creates huge, unreadable equations, we will use a numerical differentiation shortcut (finite differences) to build the Jacobian. This is an industry-standard method for complex robots.

```
import numpy as np
def forward_kinematics_3d(q, lengths):
    """
    Computes the 3D position and orientation (Roll, Pitch, Yaw) of the tip.
    Structure: Anthropomorphic 6-axis configuration.
    """
    # Quick unpacking of links and angles
    l1, l2, l3, l4 = lengths
    q1, q2, q3, q4, q5, q6 = q

    # Rotation matrices for each joint (Simplified spatial representation)
    # Base rotation around Z
    s1, c1 = np.sin(q1), np.cos(q1)
    # Shoulder, Elbow, Wrist pitch around Y
    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)
    s4, c4 = np.sin(q4), np.cos(q4)
    
    # Forward calculation of tip positions in 3D Space
    # Ground to shoulder height
    z = l1 
    # Link 2 projection
    x = l2 * c2 * c1
    y = l2 * c2 * s1
    z += l2 * s2
    
    # Link 3 projection
    x += l3 * np.cos(q2 + q3) * c1
    y += l3 * np.cos(q2 + q3) * s1
    z += l3 * np.sin(q2 + q3)
    
    # Link 4 (Hand/Tool tip) projection
    x += l4 * np.cos(q2 + q3 + q4) * c1
    y += l4 * np.cos(q2 + q3 + q4) * s1
    z += l4 * np.sin(q2 + q3 + q4)
    
    # Orientation tracking (Roll, Pitch, Yaw) simplified for demo tracking
    roll = q5
    pitch = q2 + q3 + q4
    yaw = q1 + q6
    
    return np.array([x, y, z, roll, pitch, yaw])
def numerical_jacobian_3d(q, lengths, h=1e-5):
    """
    Builds a 6x6 Jacobian matrix using numerical differentiation.
    Approximates how much each joint shifts the 3D position & orientation.
    """
    J = np.zeros((6, 6))
    f_current = forward_kinematics_3d(q, lengths)
    
    for i in range(6):
        q_perturbed = np.array(q, dtype=float)
        q_perturbed[i] += h # Nudge a single joint slightly
        
        f_perturbed = forward_kinematics_3d(q_perturbed, lengths)
        J[:, i] = (f_perturbed - f_current) / h # Record the rate of change
        
    return J
def spatial_3d_ik(target_pose, lengths, q_current, q_min, q_max, max_iter=100, tol=1e-3, damping=0.1):
    q = np.array(q_current, dtype=float)
    q_min = np.array(q_min, dtype=float)
    q_max = np.array(q_max, dtype=float)
    
    for i in range(max_iter):
        # 1. Forward Kinematics (Current 3D Pose)
        current_pose = forward_kinematics_3d(q, lengths)
        
        # 2. Pose Error Calculation (Position + Orientation)
        error = target_pose - current_pose
        
        # Wrap orientation errors to stay within [-pi, pi] bounds smoothly
        error[3:] = (error[3:] + np.pi) % (2 * np.pi) - np.pi
        
        if np.linalg.norm(error) < tol:
            print(f"Target 3D Pose reached in {i} iterations.")
            return q
            
        # 3. Compute 6x6 Numerical Jacobian
        J = numerical_jacobian_3d(q, lengths)
        
        # 4. Measure Manipulability index
        manipulability = np.abs(np.linalg.det(J))
        
        # 5. Damped Least Squares for 3D system
        I = np.eye(6)
        # Adapt damping dynamically if close to workspace edge
        lam = damping if manipulability < 0.01 else 0.0
        
        J_damped_inv = J.T @ np.linalg.inv(J @ J.T + lam**2 * I)
        dq = J_damped_inv @ error
        
        # 6. Apply limits and save step
        q = np.clip(q + dq, q_min, q_max)
        
    print("Max iterations reached. Core target estimation returned.")
    return q
# --- Execution Setup ---# 4 structural link lengths: Base-height, Upper-arm, Forearm, Hand-lengtharm_links = [0.5, 1.0, 1.0, 0.3] 
# Limits for 6 joints (-180 to 180 degrees in radians)min_limits = [np.radians(-180)] * 6max_limits = [np.radians(180)] * 6
# Desired 3D Goal: [X=1.2m, Y=0.8m, Z=1.1m, Roll=0rad, Pitch=0.5rad, Yaw=0.2rad]target_spatial_pose = np.array([1.2, 0.8, 1.1, 0.0, 0.5, 0.2])initial_guess_joints = [0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
final_3d_joints = spatial_3d_ik(target_spatial_pose, arm_links, initial_guess_joints, min_limits, max_limits)

print("\n--- Final 6-Joint Configurations ---")for idx, angle in enumerate(final_3d_joints, 1):
    print(f"Joint {idx}: {np.degrees(angle):.2f} degrees")
```

## Key Changes for 3D Space

* The 6-Dimensional Task Space: The error vector (error) tracking now monitors 6 attributes simultaneously. The tracking algorithm works until the hand is perfectly positioned in coordinates and pointing in the exact target direction.
* Numerical Differentiation: The numerical_jacobian_3d function acts like a virtual testing loop. It moves each joint by a tiny fraction (1e-5 radians) and observes how the $X, Y, Z$ and rotations react. This builds the Jacobian rows on the fly.
* Orientation Wrapping: Angles are cyclical ($360^\circ$ is identical to $0^\circ$). The line error[3:] = (error[3:] + np.pi) % (2 * np.pi) - np.pi prevents the solver from spinning the wrist joints a full circle when a minor shortcut movement is available.

Would you like to explore how to connect this 3D solver to a trajectory planner, allowing the arm to paint a smooth geometric line or circle in 3D space?

