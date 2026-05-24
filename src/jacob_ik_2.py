"""
    From `_notes/note2.md`
"""
from canvas import pg, Canvas, Vec2
import random as rnd


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

def test():
    # --- Execution Setup ---
    link1 = 1.0
    link2 = 1.0
    #  Define hard boundaries (in radians)
    # Joint 1: Can rotate 180 degrees (-90 to +90)
    # Joint 2: Restricted to positive angles only (0 to 110 degrees) to avoid bending backwards
    min_limits = [np.radians(-90), np.radians(0)]
    max_limits = [np.radians(90), np.radians(110)]
    #    Target point that would normally require Joint 2 to bend backwards to reach comfortably
    target = np.array([0.5, -0.5])
    initial_joints = [0.1, 0.2]
    final_joints = limited_adaptive_ik(target, link1, link2, initial_joints, min_limits, max_limits)

    print(f"Final joint angles: {np.degrees(final_joints)} degrees")


class Joint:
    def __init__(self,
        radius: float,
        pos: tuple[float, float],
        color: tuple[int,int,int]=(255,255,255)):
        self.radius = radius
        self.pos = pos
        self.color = color

    def draw(self, surf: pg.Surface, thick=0):
        pg.draw.circle(surf, self.color, self.pos, self.radius, width=thick)

    def __repr__(self):
        return f"Joint Object | [rad: {self.radius}] | [pos:  {self.pos}]"


class App:
    def __init__(self):
        self.canvas = Canvas("Jacobian Inverse Kinematics with Joint Adaptive Dampening and Limits",
            (1200, 675), {'update': self.update, 'render': self.render})

        self.colors = [(209, 36, 109), (127, 109, 36), (79, 79, 209)]

        self.g_rad = 15 #   global radius
        self.target_pos = None
        self.joints = []
        self.lengths = []
        self.init_joints((550, 337.5))
        self.rad_o_effect = 40
        self.min_limits = [np.radians(-90), np.radians(0)]
        self.max_limits = [np.radians(90), np.radians(110)]


    def init_joints(self, startPos: tuple[float, float]):
        length = 150
        dl = 15
        x = startPos[0]
        y = startPos[1]
        for idx in range(3):
            self.joints += [Joint(self.g_rad, (x, y), pg.Color(self.colors[idx]))]
            x += rnd.randrange(length - dl, length + dl) * np.cos(rnd.randrange(-180, 180) * (np.pi / 180))
            y += rnd.randrange(length - dl, length + dl) * np.sin(rnd.randrange(-180, 180) * (np.pi / 180))

        v1 = Vec2(self.joints[1].pos) - Vec2(self.joints[0].pos)
        v2 = Vec2(self.joints[2].pos) - Vec2(self.joints[1].pos)
        self.lengths += [v1.length()]
        self.lengths += [v2.length()]
        print("Joints: ", self.joints)
        print("Lengths: ", self.lengths)

  
    def solve_ik(self):
        v1 = Vec2(self.joints[1].pos) - Vec2(self.joints[0].pos)
        v2 = Vec2(self.joints[2].pos) - Vec2(self.joints[1].pos)

        q_current = [
            np.atan2(v1.normalize().y, v1.normalize().x),
            np.atan2(v2.normalize().y, v2.normalize().x)
        ]
         
        #   new angles/orientations

        #    vector from origin to target --- first joint is the origin
        #   this effectively makes the position on screen relative to
        #   the first joint as origin.`
        
        target_pos = Vec2(self.target_pos) - Vec2(self.joints[0].pos) 
        target_pos_dist_from_o = target_pos.length()
        direction = target_pos.normalize()
        
        #   ensure the target position does not surpass the fully extended
        #   length of the arm. If it does, the result starts to *tweak* 
        #   the 1e-2 in the condition is needed. But it's not needed in the recalculations
        if target_pos_dist_from_o > (sum(self.lengths) + 1e-2):
            target_pos = Vec2(self.joints[0].pos) + direction * (sum(self.lengths) + 1e-2)
            #   make relative to origin again
            target_pos = target_pos - Vec2(self.joints[0].pos)


        #   ensure that the least distance is reachable by the length of the arms
        if target_pos_dist_from_o < (self.lengths[0] - self.lengths[1] + 1e-2):
            target_pos = Vec2(self.joints[0].pos) + direction * (self.lengths[0] - self.lengths[1] + 5e-2)
            target_pos = target_pos - Vec2(self.joints[0].pos)
        

        new_q = limited_adaptive_ik(target_pos, self.lengths[0], self.lengths[1],
                q_current, self.min_limits, self.max_limits, max_iter=100, tol=1e-4,
                max_damping=0.3, threshold=0.05)        # 4) Finally

        #   for Joint idx 1
        x = self.lengths[0] * np.cos(new_q[0])
        y = self.lengths[0] * np.sin(new_q[0])
        self.joints[1].pos = (self.joints[0].pos[0] + x, self.joints[0].pos[1] + y)

        #   for joint idx 2 or idx -1
        x = self.lengths[0] * np.cos(new_q[0]) + self.lengths[1] * np.cos(new_q[0] + new_q[1])
        y = self.lengths[0] * np.sin(new_q[0]) + self.lengths[1] * np.sin(new_q[0] + new_q[1])
        self.joints[-1].pos = (self.joints[0].pos[0] + x, self.joints[0].pos[1] + y) 



    def sense(self):
        mouse_pressed = pg.mouse.get_pressed()
        if not mouse_pressed[0]:
            return

        mouse_pos = pg.mouse.get_pos()
        # x_dif = np.power(mouse_pos[0] - self.joints[-1].pos[0], 2)
        # y_dif = np.power(mouse_pos[1] - self.joints[-1].pos[1], 2)
        # dif = np.power(x_dif + y_dif, 0.5)


        # if dif <= self.rad_o_effect:
            # self.joints[-1].color = (255, 0, 255)
            # grad = (Vec2(mouse_pos) - Vec2(self.joints[1].pos)).normalize() 
            # self.joints[-1].pos = tuple(Vec2(self.joints[1].pos) + grad * self.lengths[-1])
            
        # self.joints[-1].color = self.colors[-1]

        self.target_pos = mouse_pos
        self.solve_ik()
        # print("Old Target Pos: ", self.target_pos)
        # print("Remapped Target Pos: ", self.remap(self.target_pos))
        # print("Unremapped Target Pos: ", self.unremap(self.remap(self.target_pos)))


    def update(self):
        self.sense()


    def render(self):
        prev_joint = None
        for idx, joint in enumerate(self.joints):
            joint.draw(self.canvas.screen)
            if idx > 0:
                prev_joint = self.joints[idx - 1]
                pg.draw.line(self.canvas.screen, (255, 255, 255),
                            prev_joint.pos, joint.pos, 3)


    def run(self):
        self.canvas.run()


if __name__ == "__main__":
    App().run()
    # test()