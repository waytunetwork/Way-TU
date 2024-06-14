import robotic as ry
import numpy as np
import time
import math

from scipy.spatial.transform import Rotation as R
import WayTu_Rai.helper_functions as hlp


class LiftTask:
    def __init__(self, C, waypoint_att):
        self.C = C

        self.waypoint_att = waypoint_att
        self.learning_rate = 0.01
    
    def create_lift_static(self):
        self.C.addFrame("plate", "table") \
            .setShape(ry.ST.ssBox, [0.2,0.4,0.04,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.4, 0.3, 0.075]) \
            .setContact(1)
        self.C.addFrame("back-leg1", "plate") \
            .setShape(ry.ST.ssBox, [0.04,0.04,0.4,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0, -0.2, 0.20]) \
            .setContact(1)
        self.C.addFrame("back-joint", "back-leg1") \
            .setShape(ry.ST.ssBox, [0.06,0.04,0.04,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.04, 0, 0.1]) \
            .setContact(1)
        self.C.addFrame("back-leg2", "back-joint") \
            .setShape(ry.ST.ssBox, [0.04,0.04,0.4,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.04, 0, -0.1]) \
            .setContact(1)
        
        self.C.addFrame("front-leg1", "plate") \
            .setShape(ry.ST.ssBox, [0.04,0.04,0.4,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0, 0.2, 0.20]) \
            .setContact(1)
        self.C.addFrame("front-joint", "front-leg1") \
            .setShape(ry.ST.ssBox, [0.06,0.04,0.04,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.04, 0, 0.1]) \
            .setContact(1)
        self.C.addFrame("front-leg2", "front-joint") \
            .setShape(ry.ST.ssBox, [0.04,0.04,0.4,0.005]) \
            .setColor([.1]) \
            .setRelativePosition([0.04, 0, -0.1]) \
            .setContact(1)
        self.obj_pos = self.C.getFrame("back-joint").getPosition() + [0, 0.2, 0.04]
        print("self.obj pose: ", self.obj_pos)
        self.obj_shape = [0.03,0.5,0.03,0.005]
        self.C.addFrame("obj") \
            .setShape(ry.ST.ssBox, self.obj_shape) \
            .setColor([0.35,0.98,1.0]) \
            .setPosition(self.obj_pos) \
            .setMass(0.00001)\
            .setContact(1)
    
    def find_lift_waypoints_static(self):
        distance_x = - 0.1
        height = 0.1
        x_bound = 0.0
        y_bound = np.random.uniform(-self.obj_shape[1]/2.5,self.obj_shape[1]/2.5)
        z_bound = np.random.uniform(-self.obj_shape[2]/2.1,self.obj_shape[2]/2.1)

        way_ready = self.C.getFrame("obj").getPosition() + [distance_x + x_bound,y_bound,  -height + z_bound]
        
        y_bound = np.random.uniform(-self.obj_shape[1]/2.5,self.obj_shape[1]/2.5)
        z_bound = np.random.uniform(-self.obj_shape[2]/2.1,self.obj_shape[2]/2.1)
        
        way_go = self.C.getFrame("obj").getPosition() + [distance_x + x_bound, y_bound, height*2 + z_bound]
        return way_ready, way_go
    
    def grasp_score(self, old_position, new_position,obj_new):
        grasp_distance = math.sqrt((new_position[0] - old_position[0]) ** 2 + (new_position[1] - old_position[1]) ** 2+ (new_position[2] - old_position[2]) ** 2) 
        change_z = new_position[2] - old_position[2]
        grasp_score = 1 if grasp_distance > 0.1  else 0 # and change_z > 0.1
        
        obj_score = math.sqrt((obj_new[0] - self.obj_pos[0]) ** 2 + (obj_new[1] - self.obj_pos[1]) ** 2+ (obj_new[2] - self.obj_pos[2]) ** 2) 

        lift_score =  0.4 * grasp_score + 0.6 * obj_score
        
        print("obj_score: ", obj_score)
        print("grasp_score: ", grasp_score)
        print("lift_score: ", lift_score)

        return lift_score, grasp_score
    
    def update_probabilities(self, selected_tool,idx, success):
        prob = self.waypoint_att.orientation_probabilities[selected_tool]
        prob[idx] = prob[idx] + success*self.learning_rate - (1 -success)*self.learning_rate
        self.waypoint_att.orientation_probabilities[selected_tool] = prob/np.sum(prob)

class LiftWaypointAttributes:
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
