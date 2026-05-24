"""
    From `_notes/note4.md`

    Demonstrates forward kinematics.
    It also turns out that the Jacobian matrix for join velocities
    is only useful in inverse kinematics.
"""
from canvas import pg, Canvas, Vec2


import numpy as np
def compute_kinematics(q: tuple[float, float], l1: float, l2: float):
    """
    Computes Forward Kinematics and the Jacobian for a 2-link planar arm.
    q: list/array of 2 joint angles [theta1, theta2] in radians
    l1 and l2 are the starts of each link arm
    """
    q1, q2 = q
    
    # 1. Forward Kinematics (Position of the tip)
    # x = l1*cos(q1) + l2*cos(q1+q2)
    # y = l1*sin(q1) + l2*sin(q1+q2)
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    pos = np.array([x, y])

    # 2. Jacobian Matrix Derivation
    # J = [[dx/dq1, dx/dq2],
    #      [dy/dq1, dy/dq2]]
    #   turns out that the jacobian is only used in inverse kinematics!
    J = np.array([
        [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
        [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2)]
    ])
    
    return pos, J

def test():
    # --- Simulation Parameters ---
    link_lengths = (1.0, 1.0)
    joint_angles = [np.pi/4, np.pi/4] # 45 and 45 degrees
    joint_velocities = np.array([0.1, 0.2]) # rad/s
    # Execute
    pos, jac = compute_kinematics(joint_angles, *link_lengths)
    tip_velocity = jac @ joint_velocities
    # Output results
    print(f"Tip Position: x={pos[0]:.3f}, y={pos[1]:.3f}")
    print(f"Jacobian Matrix:\n{jac}")
    print(f"Tip Velocity Vector (vx, vy): {tip_velocity}")


class Link:
    def __init__(self,
        startPos: tuple[float, float], endPos: tuple[float, float],
        length: float, orientation: float):
        self.start = startPos
        self.end = endPos
        self.length = length
        self.orientation = orientation #    original orientation 

    def draw(self, surf: pg.Surface, color: tuple[int, int, int]=(255,255,255), thick: int=1):
        pg.draw.line(surf, color, self.start, self.end, width=max(1, thick))

class Point:
    def __init__(self, radius: float, pos: tuple[float, float]):
        self.radius = radous
        self.pos = pos

    def draw(self, surf: pg.Surface, color:tuple[int,int,int]=(255,255,255), thick=0):
        pg.draw.circle(surf, self.color if len(color) == 0 else color,
                    self.pos, self.radius, width=thick)

class Marker:
    @classmethod
    def draw(self, surf: pg.Surface, radius:float,
            pos: tuple[int,int], color:tuple[int,int,int]=(255,255,255),
            thick:int=0, selected:bool=False):
        col =  color if len(color) == 0 else color
        col = col if not selected else (255, 255, 25)
        pg.draw.circle(surf, col,
                        pos, radius, width=thick)


class App:
    def __init__(self):
        self.canvas = Canvas("Jacobian Forward Kinematics 2 Arm",
            (1200, 675), {'update': self.update, 'render': self.render})

        self.link1: Link = self.init_link((600, 332.5), 45 * (np.pi/180), 150)
        self.link2: Link = self.init_link(self.link1.end, 45 * (np.pi/180), 150)

        self.rad_o_effect = 40

    def init_link(self, start: tuple[float, float], orientation: float, length: float):
        endx = start[0] + length * np.cos(orientation)
        endy = start[1] + length * np.sin(orientation)
        return Link(start, (endx, endy), length, orientation)

    def sense(self):
        """Using mouse to reposition the links' positions"""
        mouse_pressed = pg.mouse.get_pressed()
        if not mouse_pressed[0]:
            return
        # print(mouse_pressed)
        mouse_pos = pg.mouse.get_pos()
        #   only the start of link1 can be affected
        x_dif = np.power(mouse_pos[0] - self.link1.end[0], 2)
        y_dif = np.power(mouse_pos[1] - self.link1.end[1], 2)
        dif = np.power(x_dif + y_dif, 0.5)
        if dif <= self.rad_o_effect:
            direction = Vec2(mouse_pos) - Vec2(self.link1.start)
            direction = direction.normalize()
            self.link1.end = self.link1.start + direction * self.link1.length
            self.link2.start = self.link1.end
            return True
        return False


    def update(self):
        """
        when calculating the ratios, the adjacent side's length should really not
        be greater than the hypotenuse length
        Best to use the normalized vector form
        because actual lengths may change when mouse
        is used to reposition link joints
        which introduce estimation errors
        and affect accuracy of result when
        trying to get orientation from these
        incosistent length values

        E.gs:
            # link1
            link1_length = (l1p2 - l1p1).length()
            print("Link1 Length: ", link1_length)
            # ratio1 = min((self.link1.end[0] - self.link1.start[0]), self.link1.length)/self.link1.length
                
            # link2
            link2_length = (l2p2 - l2p1).length()
            print("Link2 Length: ", link2_length)
            # ratio2 = min((self.link2.end[0] - self.link2.start[0]), self.link2.length)/self.link2.length

        Sol1: Calculating orientation using cosine
        Shortcoming: angle is limited to 0->180

        # link1 orientation
        l1p1 = Vec2(self.link1.start)
        l1p2 = Vec2(self.link1.end)
        l1_normalized = (l1p2 - l1p1).normalize()
        link1_length = (l1p2 - l1p1).length()
        print("Link1 Length: ", link1_length)
        ratio1 = l1_normalized.x / l1_normalized.length()
        o1 = np.acos(ratio1)
        print("Angle Link1: ", o1 * (180/np.pi))

        # link2 orientation
        l2p1 = Vec2(self.link2.start)
        l2p2 = Vec2(self.link2.end)
        l2_normalized =(l2p2 - l2p1).normalize()
        link2_length = (l2p2 - l2p1).length()
        print("Link2 Length: ", link2_length)
        ratio2 = l2_normalized.x / l2_normalized.length()
        o2 = np.acos(ratio2)
        print("Angle Link2: ", o2 * (180/np.pi))

        Sol2: Calculate orientation using tan

        """

        self.joint_selected = self.sense()

        # # link1 orientation
        l1p1 = Vec2(self.link1.start)
        l1p2 = Vec2(self.link1.end)
        l1_normalized = (l1p2 - l1p1).normalize()
        link1_length = (l1p2 - l1p1).length()
        print("Link1 Length: ", link1_length)
        o1 = np.atan2(l1_normalized.y, l1_normalized.x)
        print("Angle Link1: ", o1 * (180/np.pi))

        # link2 orientation
        l2p1 = Vec2(self.link2.start)
        l2p2 = Vec2(self.link2.end)
        l2_normalized =(l2p2 - l2p1).normalize()
        link2_length = (l2p2 - l2p1).length()
        print("Link2 Length: ", link2_length)
        o2 = np.atan2(l2_normalized.y, l2_normalized.x)
        print("Angle Link2: ", o2 * (180/np.pi))

        pos, jac = compute_kinematics((o1, self.link2.orientation), self.link1.length, self.link2.length)
        print("Pos: ", pos)

        self.link2.end = (self.link1.start[0] + pos[0], self.link1.start[1] + pos[1])
        # self.link2.end = (pos[0], pos[1])
        print("Tip Final: ", self.link2.end)


    def render(self):
        self.link1.draw(self.canvas.screen, (209, 96, 30), 4)
        self.link2.draw(self.canvas.screen, (30, 96, 209), 4)
        Marker.draw(self.canvas.screen, 10, self.link1.start)
        Marker.draw(self.canvas.screen, 10, self.link1.end, selected=self.joint_selected)
        Marker.draw(self.canvas.screen, 10, self.link2.end)


    def run(self):
        self.canvas.run()

if __name__ == "__main__":
    App().run()
    # test()