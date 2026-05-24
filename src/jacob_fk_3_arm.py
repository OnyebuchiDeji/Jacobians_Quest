"""
    From `_notes/note4.md`

    It works properly. Moving different joints may result to an immediate
    change in how the links are positioned.
    It might seem wrong. But it's not wrong. It happens because the Forward
    Kinematics algorithm indeed properly runs, adding the angles for the
    subsequent Joints.

    Also, my implementation using links made it complicated. Furthermore,
    instead of how it was done here, I could have simply modified the
    `compute_kinematics` function to accommodate a three-link/arm or four-joint
    system. But I did it the way below just for the sake of it.
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
        self.color = (255, 255, 255)
        self.selected = False

    def draw(self, surf: pg.Surface, thick: int=1):
        pg.draw.line(surf, self.color, self.start, self.end, width=max(1, thick))

    def update_orientation(self):
        p1 = Vec2(self.start)
        p2 = Vec2(self.end)
        normalized = (p2 - p1).normalize()
        self.orientation = round(np.atan2(normalized.y, normalized.x), 4)
        print("Orientation Updated!", self.orientation)


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

        self.links: list[Link] = []
        self.init_arm_state((600, 332.5))
        self.rad_o_effect = 40


    def init_arm_state(self, startPos):
        """Initalizes original state of arms"""
        orientations = []
        if len(self.links) > 0:
            for link in self.links:
                #   save orientations
                orientations.append(link.orientation)
            self.links.clear()

        length = 150
        for idx in range(3):
            theta = -45 * (np.pi/180) if len(orientations) == 0 else orientations[idx]
            if idx == 0:
                self.links += [self.init_link(startPos, theta, length)]
                continue
            self.links += [self.init_link(self.links[-1].end, theta, length)]

        self.links[0].color = (209, 96, 30)
        self.links[1].color = (30, 96, 209)
        self.links[2].color = (96, 209, 30)
        

    def init_link(self, start: tuple[float, float], orientation: float, length: float):
        endx = start[0] + length * np.cos(orientation)
        endy = start[1] + length * np.sin(orientation)
        return Link(start, (endx, endy), length, orientation)

    def solve_fk(self, startIdx):
        start_link = self.links[startIdx - 1]
        # slp1 = Vec2(start_link.start)
        # slp2 = Vec2(start_link.end)
        # sl_normalized = (slp2 - slp1).normalize()
        # o1 = np.atan2(sl_normalized.y, sl_normalized.x)
        start_link.update_orientation()
        o1 = start_link.orientation

        final_pos = start_link.start

        start_length = (Vec2(start_link.end) - Vec2(start_link.start)).length()

        # print("Start Length: ", start_length)
        for idx in range(startIdx, len(self.links)):
            cl = self.links[idx] # current link
            # cl.update_orientation()

            # clp1 = Vec2(cl.start)
            # clp2 = Vec2(cl.end)
            # cl_normalized =(clp2 - clp1).normalize()
            # o2 = np.atan2(cl_normalized.y, cl_normalized.x)


            pos, jac = compute_kinematics((o1, cl.orientation),
                         start_length, cl.length)

            # final_pos = (final_pos[0] + pos[0], final_pos[1] + pos[1])
            final_pos = (start_link.start[0] + pos[0], start_link.start[1] + pos[1])

            self.links[idx].end = final_pos
            # self.links[idx].update_orientation()
            if idx + 1 < len(self.links):
                self.links[idx + 1].start = final_pos
                # self.links[idx + 1].update_orientation()

            #   Recalculate new length to use in kinematics operation
            #   from the recently solved point for idx1
            start_length = (Vec2(self.links[idx].end) - Vec2(start_link.start)).length()            
            slp1 = Vec2(start_link.start)
            slp2 = Vec2(self.links[idx].end)
            sl_normalized = (slp2 - slp1).normalize()
            o1 = np.atan2(sl_normalized.y, sl_normalized.x)
            # final_pos = start_link.start
            # if idx + 1 < len(self.links):
            #     o1 = self.links[idx + 1].orientation
            # print("After Orientation")
            # return
            if idx == 2:
                nl = (Vec2(self.links[idx].end) - Vec2(cl.start)).length()
                print("Last Link Length: ", nl)
            




    def sense(self):
        """Using mouse to reposition the links' positions"""
        mouse_pressed = pg.mouse.get_pressed()
        if not mouse_pressed[0]:
            return

        def solve_pos(idx, mouse_pos, link):
            prev_link = self.links[idx - 1]
            direction = (Vec2(mouse_pos) - Vec2(prev_link.start)).normalize()
            prev_link.end = Vec2(prev_link.start) + direction * link.length
            link.start = prev_link.end 
            prev_link.update_orientation()
            link.selected = True

        mouse_pos = pg.mouse.get_pos()
        for idx, link in enumerate(self.links):
            #   only the start of link1 can be affected
            x_dif = np.power(mouse_pos[0] - link.start[0], 2)
            y_dif = np.power(mouse_pos[1] - link.start[1], 2)
            dif = np.power(x_dif + y_dif, 0.5)
            link.selected = False
            if dif <= self.rad_o_effect:
                print("Selected Idx: ", idx)
                match idx:
                    case 0:
                        self.init_arm_state(mouse_pos)
                        link.selected = True
                        return
                    case _:
                        solve_pos(idx, mouse_pos, link)
                        self.solve_fk(idx)
                        return
            link.selected = False


    def update(self):
        self.sense()

        
        # link1 orientation
        # l1p1 = Vec2(self.link1.start)
        # l1p2 = Vec2(self.link1.end)
        # l1_normalized = (l1p2 - l1p1).normalize()
        # link1_length = (l1p2 - l1p1).length()
        # print("Link1 Length: ", link1_length)
        # o1 = np.atan2(l1_normalized.y, l1_normalized.x)
        # print("Angle Link1: ", o1 * (180/np.pi))

        # # link2 orientation
        # l2p1 = Vec2(self.link2.start)
        # l2p2 = Vec2(self.link2.end)
        # l2_normalized =(l2p2 - l2p1).normalize()
        # link2_length = (l2p2 - l2p1).length()
        # print("Link2 Length: ", link2_length)
        # o2 = np.atan2(l2_normalized.y, l2_normalized.x)
        # print("Angle Link2: ", o2 * (180/np.pi))

        # pos, jac = compute_kinematics((o1, self.link2.orientation), self.link1.length, self.link2.length)
        # print("Pos: ", pos)

        # self.link2.end = (self.link1.start[0] + pos[0], self.link1.start[1] + pos[1])
        # # self.link2.end = (pos[0], pos[1])
        # print("Tip Final: ", self.link2.end)


    def render(self):
        for link in self.links:
            link.draw(self.canvas.screen, thick=4)
            Marker.draw(self.canvas.screen, 10, link.start, selected=link.selected)

        #   tip
        Marker.draw(self.canvas.screen, 10, self.links[-1].end)


    def run(self):
        self.canvas.run()

if __name__ == "__main__":
    App().run()
    # test()