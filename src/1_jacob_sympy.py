"""
    From `_notes/note1.md`
"""
print("This is `1_jacob_sympy.py`")


import sympy as sp
import numpy as np

def get_2dof_jacobian():
    # Define symbolic variables
    t1, t2 = sp.symbols('theta1 theta2')
    L1, L2 = sp.symbols('L1 L2')

    # Forward Kinematics equations
    x = L1 * sp.cos(t1) + L2 * sp.cos(t1 + t2)
    y = L1 * sp.sin(t1) + L2 * sp.sin(t1 + t2)

    # Compute Jacobian Matrix (partial derivatives)
    funcs = sp.Matrix([x, y])
    vars = sp.Matrix([t1, t2])
    J = funcs.jacobian(vars)
    
    return J, (t1, t2, L1, L2)
# Example usage
J_sym, symbols = get_2dof_jacobian()
t1_sym, t2_sym, L1_sym, L2_sym = symbols
# Evaluate at a specific configuration: L1=1, L2=1, theta1=45deg, theta2=0 deg
eval_params = {L1_sym: 1.0, L2_sym: 1.0, t1_sym: np.pi/4, t2_sym: 0}
J_numeric = J_sym.subs(eval_params).evalf()

print("Symbolic Jacobian:")
sp.pprint(J_sym)
print("\nNumerical Jacobian at (45°, 0°):")
print(np.array(J_numeric).astype(np.float64))
