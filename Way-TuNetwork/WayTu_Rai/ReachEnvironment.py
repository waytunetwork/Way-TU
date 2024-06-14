import robotic as ry
import numpy as np
import time
import math

from scipy.spatial.transform import Rotation as R
import WayTu_Rai.helper_functions as hlp


class ReachTask:
    def __init__(self, C, side, waypoint_att):
        self.side = side
        self.C = C
        self.waypoint_att = waypoint_att
        self.learning_rate = 0.01

        self.reach_random()

    def create_new_reach_static(self):

        self.C.addFrame("down-wall", "table") \
            .setShape(ry.ST.ssBox, [0.2,0.6,0.4,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.4, 0.30, 0.2]) \
            .setContact(1)
            
        self.C.addFrame("left-wall", "down-wall") \
            .setShape(ry.ST.ssBox, [0.2, 0.25, 0.1,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0, 0.175, 0.25]) \
            .setContact(1)
        self.C.addFrame("right-wall", "down-wall") \
            .setShape(ry.ST.ssBox, [0.2, 0.25, 0.1,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0, -0.175, 0.25]) \
            .setContact(1)
        
        self.obj_pos = self.C.getFrame("down-wall").getPosition() + [0, 0., 0.25] 
        self.obj_shape = [0.19, 0.09, 0.09, 0.005]
        
        self.C.addFrame('obj')\
            .setPosition(self.obj_pos) \
            .setShape(ry.ST.ssBox,self.obj_shape )\
            .setColor([0.35,0.98,1.0]) \
            .setMass(0.00001)\
            .setContact(1)
    
    def score(self, new_position):
        return math.sqrt((new_position[0] - self.obj_pos[0]) ** 2 + (new_position[1] - self.obj_pos[1]) ** 2+ (new_position[2] - self.obj_pos[2]) ** 2) 
    

    def update_probabilities(self, selected_tool, idx, success):
        prob = self.waypoint_att.orientation_probabilities[selected_tool]
        prob[idx] = prob[idx] + success*self.learning_rate - (1 -success)*self.learning_rate
        self.waypoint_att.orientation_probabilities[selected_tool] = prob/np.sum(prob)
        print("New Prob: ", self.waypoint_att.orientation_probabilities[selected_tool])


    def grasp_score(self, old_position, new_position, obj_new):
        grasp_distance = math.sqrt((new_position[0] - old_position[0]) ** 2 + (new_position[1] - old_position[1]) ** 2+ (new_position[2] - old_position[2]) ** 2) 
        obj_score = math.sqrt((obj_new[0] - self.obj_pos[0]) ** 2 + (obj_new[1] - self.obj_pos[1]) ** 2+ (obj_new[2] - self.obj_pos[2]) ** 2) 
        change_z = new_position[2] - old_position[2]
        grasp_score = 1 if grasp_distance > 0.1  else 0 # and change_z > 0.1
        reach_score = 0.4 * grasp_score + 0.6 * obj_score
        return reach_score, grasp_score

    def find_reach_object_waypoints_static(self):
        distance = 0.3
        y_bound = np.random.uniform(-self.obj_shape[1]/2.1,self.obj_shape[1]/2.1)
        z_bound = np.random.uniform(-self.obj_shape[2]/2.1,self.obj_shape[2]/2.1)

        way_ready = self.C.getFrame("obj").getPosition() - [distance, y_bound, z_bound]
        way_go = self.C.getFrame("obj").getPosition() + [distance, y_bound, z_bound]
        return way_ready, way_go
        
        
    def find_reach_object_waypoints(self):
        distance = -0.4
        wall1 = self.C.getFrame("wall-1").getPosition()
        wall2 = self.C.getFrame("wall-2").getPosition()
        ball = self.C.getFrame("ball").getPosition()

        wall_vector = wall1 - wall2
        unit_wall_vector = wall_vector/np.linalg.norm(wall_vector)

        perpendicular_vector = np.array([-unit_wall_vector[1], unit_wall_vector[0]])
        
        way_ready = ball[:2]+ distance * perpendicular_vector
        way_ready = np.append(way_ready, ball[2])

        way_go = ball[:2]- distance * perpendicular_vector
        way_go = np.append(way_go, ball[2])

        return way_ready, way_go
    
    def reach_object_rotation(self,object_rotation): 
        rotation_quaternion = R.from_euler('x', 180, degrees=True).as_quat()
        rotated_quaternion = R.from_quat(object_rotation) * R.from_quat(rotation_quaternion)
        rotated_quaternion = rotated_quaternion.as_quat()

        return rotated_quaternion



    
        
class ReachWaypointAttributes:
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