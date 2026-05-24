####  Date: 12-03-2026

#  Jacobian Matrices in Kinematics

### Use of Jacobian Matrices in 2D and 3D Kinematics


## Note 2

####  Me
Yes, I would like for you to demonstrate python examples that use the matrix for inverse kinematics and singularity detection, and afterwards, give your suggestions: what combinations of modules is best for mathematical operations such as these? Compare and give your recommendations.

####   Response from Google AI

Building on the Jacobian derivation, here is how you can use that matrix for practical robot control and safety.
1. Inverse Kinematics (Numerical Method)
To move the end-effector to a target, we don't always have a simple formula. Instead, we use the Jacobian Inverse (or Pseudoinverse) to iteratively "nudge" the joints until the tip reaches the goal. [1, 2] 

```
import numpy as np
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
2. Singularity Detection
A singularity occurs when the Jacobian matrix loses rank, meaning the robot "loses" a direction of motion. For square matrices (like our 2-DOF arm), this happens when the determinant is zero. [3, 4, 5, 6] 
```
def check_singularity(q, l1, l2):
    # Determinant of the 2-link Jacobian simplifies to: l1 * l2 * sin(theta2)
    # Singularity occurs when sin(theta2) = 0 (arm is fully stretched or folded)
    det = l1 * l2 * np.sin(q[1])
    
    if np.abs(det) < 1e-3:
        return True, det
    return False, det
# Example: Arm fully extended (theta2 = 0)is_singular, val = check_singularity([0.5, 0.0], 1.0, 1.0)
print(f"Is Singular? {is_singular} (Det: {val})")
```

------------------------------
Comparison of Python Modules for Kinematics
When choosing a toolkit, the "best" combination depends on whether you are designing the robot or controlling it in real-time. [7, 8] 

| Module [8, 9, 10, 11, 12, 13] | Best For... | Pros | Cons |
|---|---|---|---|
| NumPy | Real-time Execution | Extremely fast; industry standard for matrix math. | Manual derivation of derivatives required. |
| SymPy | Initial Research | Handles symbolic algebra; generates the Jacobian automatically. | Too slow for high-frequency control loops. |
| Pinocchio | Production Robotics | Optimized C++ backend; handles complex 3D dynamics and derivatives. | Steeper learning curve; overkill for simple 2D arms. |
| SciPy | Optimization | Includes advanced solvers (like least_squares) for complex IK. | Heavier than NumPy for simple operations. |

My Recommendation

* For Learning/Prototyping: Use SymPy to derive your Jacobian equations once, then use NumPy to run them numerically in your main loop.
* For Professional 3D Robotics: Use Pinocchio paired with NumPy. It is widely regarded as one of the fastest libraries for computing robot dynamics and analytical derivatives. [12, 13] 

Would you like to explore Damped Least Squares, which helps robots "gracefully" handle the singularities we just detected?

[1] [https://eureka.patsnap.com](https://eureka.patsnap.com/article/implementing-inverse-kinematics-using-python-for-manipulators)
[2] [https://eureka.patsnap.com](https://eureka.patsnap.com/article/implementing-inverse-kinematics-using-python-for-manipulators)
[3] [https://www.sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0094114X22004694)
[4] [https://atmos.colostate.edu](https://atmos.colostate.edu/~aam/pdf/conferences/204.pdf#:~:text=1%20INTRODUCTION.%20A%20robot%20singular%20configuration%20is,ally%20called%20singularities%20%28Baker%20and%20Wampler%2C%201988%29.)
[5] [https://mecademic.com](https://mecademic.com/insights/academic-tutorials/what-are-singularities-6-axis-robot-arm/#:~:text=A%20robot%20singularity%20is%20a%20configuration%20in,serial%20robot%2C%20or%20serial%20manipulator%29%20has%20singularities.)
[6] [https://wanxinjin.github.io](https://wanxinjin.github.io/asu-robotics/lec13/singularity.html)
[7] [https://eureka.patsnap.com](https://eureka.patsnap.com/article/python-vs-c-for-robotic-programming-pros-and-cons#:~:text=Python%2C%20while%20slower%2C%20is%20often%20%22fast%20enough%22,which%20are%20optimized%20for%20high%2Dperformance%20scientific%20computing.)
[8] [https://thinkrobotics.com](https://thinkrobotics.com/blogs/learn/python-robot-programming-a-comprehensive-guide#:~:text=Key%20Libraries%20for%20Python%20Robotics%20*%20ROSpy,for%20complex%20mathematical%20computations%20required%20in%20robotics.)
[9] [https://www.mygreatlearning.com](https://www.mygreatlearning.com/blog/open-source-python-libraries/#:~:text=Table_title:%20Top%2030%20Python%20Libraries%20List%20Table_content:,%7C%20Primary%20Use%20Case:%20Scientific%20Computing%20%7C)
[10] [https://www.coursera.org](https://www.coursera.org/articles/python-libraries-for-data-analysis#:~:text=Some%20of%20the%20most%20popular%20libraries%20that,you%20can%20use%20them%20for%20data%20analysis.)
[11] [https://towardsdatascience.com](https://towardsdatascience.com/best-library-to-simplify-math-for-machine-learning-ed64cbe536ac/)
[12] [https://gepettoweb.laas.fr](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/)
[13] [https://sites.google.com](https://sites.google.com/site/xinsongyan/blogs/robotics/modeling-tools)
