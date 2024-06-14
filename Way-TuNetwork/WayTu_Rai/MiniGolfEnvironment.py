import robotic as ry
import numpy as np
import time
import math

from scipy.spatial.transform import Rotation as R
import WayTu_Rai.helper_functions as hlp



class MiniGolfTask:
    def __init__(self,C,waypoint_att):
        self.C = C

        self.environment_shapes = {
            "big-area":[0.3,0.4,0.1,0.005],
            "left-area": [0.05,0.1,0.1,0.005],
            "right-area": [0.05,0.1,0.1,0.005],
            "end-area": [0.3,0.1,0.1,0.005]

        }

        self.waypoint_att = waypoint_att
        self.learning_rate = 0.01

        self.obj = [
            hlp.quaternion_rotation([0,0,0,1],0,(0,0,1)),
            hlp.quaternion_rotation([0,0,0,1],90,(0,0,1)),
            hlp.quaternion_rotation([0,0,0,1],180,(0,0,1)),
            hlp.quaternion_rotation([0,0,0,1],270,(0,0,1)),
        ]
    
    def create_minigolf_static(self):

        self.C.addFrame("big-area", "table") \
            .setShape(ry.ST.ssBox, self.environment_shapes["big-area"]) \
            .setColor([.1]) \
            .setRelativePosition([0.42, 0.30, 0.05]) \
            .setContact(1)
        self.C.addFrame("left-area", "big-area") \
            .setShape(ry.ST.ssBox, self.environment_shapes["left-area"]) \
            .setColor([0.1]) \
            .setRelativePosition([0.125, -0.25, 0]) \
            .setContact(1)
        self.C.addFrame("right-area", "big-area") \
            .setShape(ry.ST.ssBox, self.environment_shapes["right-area"]) \
            .setColor([.1]) \
            .setRelativePosition([-0.125, -0.25, 0]) \
            .setContact(1)
        self.C.addFrame("end-area", "big-area") \
            .setShape(ry.ST.ssBox, self.environment_shapes["end-area"]) \
            .setColor([.1]) \
            .setRelativePosition([0, -0.35, 0]) \
            .setContact(1)
        
        self.cup_shape, self.cup_center = self.locate_cup()
        
        self.obj_pos = self.C.getFrame("big-area").getPosition()+[0,0.1,0.08]
        self.obj_shape = [0.05, 0.05, 0.05, 0.02]

        self.C.addFrame('obj')\
            .setPosition(self.obj_pos) \
            .setShape(ry.ST.ssBox, self.obj_shape)\
            .setColor([0.35,0.98,1.0]) \
            .setMass(0.00001)\
            .setContact(1)
    
    def grasp_static(self):
        ...
    
    def update_probabilities(self, selected_tool,idx, success):
        prob = self.waypoint_att.orientation_probabilities[selected_tool]
        prob[idx] = prob[idx] + success*self.learning_rate - (1 -success)*self.learning_rate
        self.waypoint_att.orientation_probabilities[selected_tool] = prob/np.sum(prob)

    def locate_cup(self):
        shape = [
            self.environment_shapes["big-area"][0] - 2 * self.environment_shapes["left-area"][0],
            self.environment_shapes["left-area"][1],
            self.environment_shapes["left-area"][2],
            0.005
        ]
        center = [
            self.C.getFrame("left-area").getPosition()[0]- 0.5*self.environment_shapes["left-area"][0] - 0.5*shape[0],
            self.C.getFrame("left-area").getPosition()[1],
            self.C.getFrame("left-area").getPosition()[2] 
        ]
        return shape, center
    
    def obj_in_cup(self, obj_center):
        x, y, z = obj_center
        hx, hy, hz = self.cup_center
        hw, hh, hd, _ = self.cup_shape

        if (hx - hw/2 <= x <= hx + hw/2) and (hy - hh/2 <= y <= hy + hh/2) and (hz - hd/2 <= z <= hz + hd/2):
            return True
        return False
        
    
    def grasp_score(self, old_position, new_position, obj_new):
        grasp_distance = math.sqrt((new_position[0] - old_position[0]) ** 2 + (new_position[1] - old_position[1]) ** 2+ (new_position[2] - old_position[2]) ** 2) 
        change_x = new_position[0] - old_position[0]
        print("Grasp-distance: ", grasp_distance)
        grasp_score = 1 if grasp_distance > 0.1 and change_x > 0.1 else 0

        obj_score = math.sqrt((obj_new[0] - self.obj_pos[0]) ** 2 + (obj_new[1] - self.obj_pos[1]) ** 2+ (obj_new[2] - self.obj_pos[2]) ** 2) 
        cup_score = 1 if self.obj_in_cup(obj_new) else 0

        minigolf_score = 0.4 * grasp_score + 0.6 * cup_score
        if self.obj_in_cup(obj_new) == False:
            cup_distance = math.sqrt((new_position[0] - self.cup_center[0]) ** 2 + (new_position[1] - self.cup_center[1]) ** 2+ (new_position[2] - self.cup_center[2]) ** 2) 
            minigolf_score = minigolf_score - 0.6 * cup_distance
       #  minigolf_score = 0.4 * grasp_score + 0.4 * cup_score + 0.2*obj_score
        gripper_last = self.C.getFrame("l_gripper").getPosition()
        dist_obj_gripper = math.sqrt((new_position[0] - gripper_last[0]) ** 2 + (new_position[1] - gripper_last[1]) ** 2+ (new_position[2] - gripper_last[2]) ** 2) 
        if dist_obj_gripper >= 0.1:
            minigolf_score -= 0.6*dist_obj_gripper
        
        print(dist_obj_gripper)

        print(f"grasp score: {grasp_score}")
        print(f"cup score: {self.obj_in_cup(obj_new)}")
        print(f"obj_score: {obj_score}")
        print(f"minigolf_score: {minigolf_score}")


        return minigolf_score, grasp_score

    
    def find_minigolf_waypoints_static(self, tool_shape):
        distance = 0.1
        height = 0.1

        z_bound = np.random.uniform(0,max(tool_shape)*0.1)

        way_ready = self.C.getFrame("obj").getPosition() + [0,distance*1.5,  z_bound]
        way_go = self.C.getFrame("obj").getPosition() + [0,-distance*5, z_bound]
        return way_ready, way_go
    
    def score(self, old_position, new_position, obj_new):
        grasp_distance = math.sqrt((new_position[0] - old_position[0]) ** 2 + (new_position[1] - old_position[1]) ** 2+ (new_position[2] - old_position[2]) ** 2) 
        obj_score = math.sqrt((obj_new[0] - self.obj_pos[0]) ** 2 + (obj_new[1] - self.obj_pos[1]) ** 2+ (obj_new[2] - self.obj_pos[2]) ** 2) 
     

class MiniGolfWaypointAttributes:
    def __init__(self):
        self.orientation_probabilities = {}
        self.initialize_orientation_probabilities()


    def initialize_orientation_probabilities(self):
        self.orientation_probabilities['tool-handle-0'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-handle-1'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-head-1'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-handle-2'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-head-2'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-handle-3'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-head-3'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-handle-4'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-handle-5'] = np.array([0.25, 0.25, 0.25, 0.25])
        self.orientation_probabilities['tool-head-5'] = np.array([0.25, 0.25, 0.25, 0.25])