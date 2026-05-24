"""
    From `_notes/note2.md`
"""
from canvas import pg, Canvas, Vec2
import random as rnd



import numpy as np
def jacobian_ik(target_pos, l1, l2, q_current, max_iter=100, tol=1e-4):
    q = np.array(q_current, dtype=float)
    for i in range(max_iter):
        # 1. Forward Kinematics: Current Tip Position
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        current_pos = np.array([x, y])
        # print("Jacobian IK Current Pos:", current_pos)
        
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

def test_ik1():
    # Usage
    target = np.array([1.2, 0.8])
    final_joints = jacobian_ik(target, 1.0, 1.0, [0.1, 0.1])
    print(f"Target reached at joints: {np.degrees(final_joints)} degrees")


def jacobian_damped_ik_2(target_pos, l1, l2, q_current, max_iter=100, tol=1e-4, damping=0.1):
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

def test_ik2():
    # Execution Matrix
    target = np.array([2.0, 0.0])
    # Target lies exactly on the workspace boundary (Singularity)
    initial_joints = [0.1, 0.1]
    final_joints = jacobian_damped_ik_2(target, 1.0, 1.0, initial_joints)

    print(f"Final joint angles: {np.degrees(final_joints)} degrees")


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

def test_ik3():
    # Execution Matrix (Targeting a point exactly on the physical workspace limit)
    target = np.array([2.0, 0.0])
    initial_joints = [0.1, 0.01]
    # Very close to fully extended straight arm
    final_joints = adaptive_jacobian_ik(target, 1.0, 1.0, initial_joints)

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
        self.canvas = Canvas("Jacobian Inverse Kinematics with Joint Adaptive Dampening",
            (1200, 675), {'update': self.update, 'render': self.render})

        self.colors = [(209, 36, 109), (127, 109, 36), (79, 79, 209)]

        self.g_rad = 15 #   global radius
        self.target_pos = None
        self.joints = []
        self.lengths = []
        self.init_joints((550, 337.5))
        self.rad_o_effect = 40


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

    def remap(self, pos: tuple[float, float]):
        """
        This method takes a point tuple existing within Python's
        coordinate system, and remaps it to a coordinate system
        where the origin is in the center 
        """
        ax0, ay0 = 0, 0
        ax1, ay1 = self.canvas.win_size

        bx0, by0 = -self.canvas.win_size[0] // 2, -self.canvas.win_size[1] // 2 
        bx1, by1 = self.canvas.win_size[0] // 2, self.canvas.win_size[1] // 2

        rx = (pos[0] - ax0) / (ax1 - ax0)
        o_x = bx0 + rx * (bx1 - bx0)

        pos_y = ay1 - pos[1] #  flip y axis
        # pos_y = pos[1]

        ry = (pos_y - ay0) / (ay1 - ay0) 
        o_y = by0 + ry * (by1 - by0)

        return (o_x, o_y)

    def unremap(self, pos: tuple[float, float]):
        bx0, by0 = 0, 0
        bx1, by1 = self.canvas.win_size

        ax0, ay0 = -self.canvas.win_size[0] // 2, -self.canvas.win_size[1] // 2 
        ax1, ay1 = self.canvas.win_size[0] // 2, self.canvas.win_size[1] // 2

        rx = (pos[0] - ax0) / (ax1 - ax0)
        o_x = bx0 + rx * (bx1 - bx0)

        pos_y = pos[1]

        ry = (pos_y - ay0) / (ay1 - ay0) 
        o_y = by0 + ry * (by1 - by0)

        o_y = by1 - o_y #  flip y axis
        return (o_x, o_y)



    def solve_ik(self):
        """
            Because the original `jacobian_ik` made its calculation
            assuming that the links where at the origin 0, I was trying
            to wrap my head around it and hence the numerous commented-out
            code below.
            At first I did not realize that the calculations were assuming
            that the vectors were raltive to the origin --- 1) Relative to origin?

            Next, after realizing this, I went to remap the whole screen space.
            It did not work --- 2) Remapped Screen Space

            I then realized that the origin is actually the first Joint but still
            used remap --- 3) Remapped Screen Space 2

            Then I removed remap altogether and just made the target point relative
            to the first joint by subtracting their vectors --- 4) Finally
        """
        # target_pos = self.joints[-1].pos
        # if self.target_pos: 
        #   vector from origin (joint 0) to joint 1
        v01 = Vec2(self.joints[1].pos) - Vec2(self.joints[0].pos)
        #   vector from joint 1 to joint 2
        v12 = Vec2(self.joints[2].pos) - Vec2(self.joints[1].pos)
        # lengths = [
        #     v1.length(),
        #     v2.length()
        # ]
        q_current = [
            np.atan2(v01.normalize().y, v01.normalize().x),
            np.atan2(v12.normalize().y, v12.normalize().x)
        ]
         
        #   new angles/orientations
        # print("Solve IK Lengths: ", self.lengths)
        # print("Solve IK Q Current: ", list(map(lambda x: np.rad2deg(x), q_current)))
        #    vector from origin to target --- first joint is the origin
        #   this effectively makes the position on screen relative to
        #   the first joint as origin.`
        # vect_o_t = Vec2(target_pos) - Vec2(self.joints[0].pos)
        # print("Before Relation: ", self.target_pos)
        target_pos = Vec2(self.target_pos) - Vec2(self.joints[0].pos) 
        target_pos_dist_from_o = target_pos.length() #  or Full Length
        direction = target_pos.normalize()
        # print("Old Target Pos: ", target_pos)

        #   ensure the target position does not surpass the fully extended
        #   length of the arm. If it does, the result starts to *tweak* 
        #   It makes sure that the target_pos has an upper limit that doesn't
        #   make the inverse kinematics calculation find a solution
        if target_pos_dist_from_o > (sum(self.lengths) + 1e-2):
            print("Full Length Touched!")
            target_pos = Vec2(self.joints[0].pos) + direction * (sum(self.lengths) + 1e-2)
            # print("Intermediate Target Pos: ", target_pos)
            target_pos = target_pos - Vec2(self.joints[0].pos)
            # print("New Target Pos: ", target_pos)


        #   THIS WAS NOT THE RIGHT SOLUTION TO THE ISSUE
        #   THE BELOW NEWER SOLUTION SOLVES IT
        #   also ensure that the point's distance from the origin is at
        #   least greater than the length of the first arm --- also to prevent tweaking
        # if target_pos_dist_from_o < (self.lengths[0] + 1e-2):
           # print("Least Length Touched!")
           #  target_pos = Vec2(self.joints[0].pos) + direction * (self.lengths[0] + 1e-2)
           #  print("Intermediate Target Pos: ", target_pos)
           #  target_pos = target_pos - Vec2(self.joints[0].pos)
           #  print("New Target Pos: ", target_pos)


        #   The issue is a case that occurs when the first arm is longer than the second

        #   also ensure that the target point can be reached using the path along two arms'/links'
        #   vectors --- this is used to check if target points very close to the origin (first Joint)
        #   can indeed be reached by the arms.
        #   it does this by checking if the target vector's distance from the origin
        #   is less than the smallest possible distance the arms combined can reach,
        #   (which is gotten by the differnce between the first arm's length and the second's
        #   if it is, it modifies the target position so that that distance from the origin
        #   is at least this smallest possible distance and then gets the appropriate vector
        #   --- it also prevents tweaking of the ik algo

        if target_pos_dist_from_o < (self.lengths[0] - self.lengths[1] + 1e-2):
            print("Least Length Touched!")
            #   ensure that the least distance is reachable by the length of the arms
            target_pos = Vec2(self.joints[0].pos) + direction * (self.lengths[0] - self.lengths[1] + 5e-2)
            print("Intermediate Target Pos: ", target_pos)
            target_pos = target_pos - Vec2(self.joints[0].pos)
            print("New Target Pos: ", target_pos)

        #   2) Remapped Screen Space
        # new_q = jacobian_ik(self.remap(target_pos), self.lengths[0], self.lengths[1], q_current, max_iter=100, tol=1e-4)

        #   Correct: First Jacobian --- naive to singularities
        # new_q = jacobian_ik(target_pos, self.lengths[0], self.lengths[1], q_current, max_iter=100, tol=1e-4)

        #   ik with singularity detection and dampening 
        # new_q = jacobian_damped_ik_2(target_pos, self.lengths[0], self.lengths[1], q_current, max_iter=100, tol=1e-4, damping=0.1)

        #   ik with singularity detection and adaptive dampening
        new_q = adaptive_jacobian_ik(target_pos, self.lengths[0], self.lengths[1],
                q_current, max_iter=100, tol=1e-4, max_damping=0.3, threshold=0.05)


        # print("new_q: ", new_q)

        #   update the joint positions using new orientations

        ##  1) Relative to origin?
        # angle = 0
        # length = 0
        # for idx, joint in enumerate(self.joints):
        #     if idx == 0: continue
        #     # if idx == len(self.joints)-1: break
        #     prev_joint_pos = self.joints[idx-1].pos
        #     angle += new_q[idx - 1]
        #     length = self.lengths[idx-1]
        #     new_joint_pos_x = prev_joint_pos[0] + length * np.cos(angle)
        #     new_joint_pos_y = prev_joint_pos[1] + length * np.sin(angle)
        #     self.joints[idx].pos = (new_joint_pos_x, new_joint_pos_y)

        # prev_joint_pos = self.joints[0].pos
        # new_joint_pos_x = prev_joint_pos[0] + self.lengths[0] * np.cos(new_q[0] * (np.pi/180))
        # new_joint_pos_y = prev_joint_pos[1] + self.lengths[0] * np.sin(new_q[0] * (np.pi/180))
        # self.joints[1].pos = (new_joint_pos_x, new_joint_pos_y)

        # 2) Remapped Screen Space
        #   for Joint idx 1
        # x = self.lengths[0] * np.cos(new_q[0])
        # y = self.lengths[0] * np.sin(new_q[0])
        # print("Joint Idx 1 (x, y): ", x, y)
        # print("Joint Idx 1 (x, y) Unremapped: ", self.unremap((x, y)))
        # self.joints[1].pos = self.unremap((x, y)) #(self.joints[0].pos[0] + x, self.joints[0].pos[1] + y) 

        # #   for joint idx 2 or idx -1
        # x = self.lengths[0] * np.cos(new_q[0]) + self.lengths[1] * np.cos(new_q[0] + new_q[1])
        # y = self.lengths[0] * np.sin(new_q[0]) + self.lengths[1] * np.sin(new_q[0] + new_q[1])
        # # current_pos = np.array([x, y])
        # print("Joint Idx 2 (x, y): ", x, y)
        # print("Joint Idx 2 (x, y) Unremapped: ", self.unremap((x, y)))
        # self.joints[-1].pos = self.unremap((x, y)) #(self.joints[1].pos[0] + x, self.joints[1].pos[1] + y) 
        # print("Mouse Pos: ", pg.mouse.get_pos())


        # 3) Remapped Screen Space 2
        # #   for Joint idx 1
        # x = self.lengths[0] * np.cos(new_q[0])
        # y = self.lengths[0] * np.sin(new_q[0])
        # x,y = unremapped = self.unremap((x, y))
        # print("Joint Idx 1 (x, y): ", x, y)
        # print("Joint Idx 1 (x, y) Unremapped: ",  unremapped)
        
        # self.joints[1].pos = (self.joints[0].pos[0] + x, self.joints[0].pos[1] + y) 

        # #   for joint idx 2 or idx -1
        # x = self.lengths[0] * np.cos(new_q[0]) + self.lengths[1] * np.cos(new_q[0] + new_q[1])
        # y = self.lengths[0] * np.sin(new_q[0]) + self.lengths[1] * np.sin(new_q[0] + new_q[1])
        # x,y = unremapped = self.unremap((x, y))
        # # current_pos = np.array([x, y])
        # print("Joint Idx 2 (x, y): ", x, y)
        # print("Joint Idx 2 (x, y) Unremapped: ", unremapped)
        # self.joints[-1].pos = (self.joints[0].pos[0] + x, self.joints[0].pos[1] + y) 
        # # print("Mouse Pos: ", pg.mouse.get_pos())

        # 4) Finally
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