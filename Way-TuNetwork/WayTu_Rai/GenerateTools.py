import robotic as ry
import numpy as np
import time
import random

import WayTu_Rai.helper_functions as hlp
import WayTu_Model.helpers as helper


class CreateTools:
    def __init__(self, C, tool_attributes, cfg, i = -1):
        self.C = C
        self.cfg = cfg

        self.tool_attributes = tool_attributes

        self.grasp_choices = {}

        self.test_tools = []
        

        distance_x = 0.23
        distance_y = 0.35
        photo = 0.0
        base = [- 0.1, 0.20, 0.055]
        self.base = base
        self.distance = distance_x

        self.learning_rate = 0.01

        self.tool_shapes = {}

        self.tool_loc  = [[base[0], base[1] + photo, 0.05],
                                   [base[0] - distance_x, base[1]+ photo, 0.055],
                                   [base[0] + distance_x, base[1] + photo, 0.055],
                                   [base[0], base[1] + distance_y, 0.057],
                                   [base[0] - distance_x, base[1] + distance_y, 0.057],
                                   [base[0] + distance_x, base[1] + distance_y, 0.057]]

        self.create_functions = [self.create_stick,
                                self.create_hammer,
                                self.create_plunger,
                                self.create_L_ruler,
                                self.create_thick_stick,
                                self.create_spatula,
                                ]
        
        self.tool_list = list(helper.get_combination(tool_list=self.create_functions, num_tools=cfg['num-tools']))    

        
        self.orientations = [
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],0),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],90),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],180),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],270)
        ]
        self.obj = [
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],90),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],90),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],270),
            hlp.quaternion_rotation_on_z_axis([0,0,0,1],270)
        ]

    def update_C(self, C):
        self.C = C

    def create_stick(self, loc):
        shape_handle = self.add_randomness([0.04, 0.25, 0.03, 0.005])
        self.tool_shapes["tool-handle-0"] = shape_handle

        self.C.addFrame("tool-handle-0", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-0"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setContact(1) \
            .setMass(.1)
        
        self.grasp_choices['tool0'] = ["tool-handle-0"]
        self.test_tools.append("tool-handle-0")
        

    def create_hammer(self, loc):

        shape_handle = self.add_randomness([0.04, 0.25, 0.03, 0.005])
        shape_head = self.add_randomness([0.08, 0.04, 0.03, 0.005])

        self.tool_shapes["tool-handle-1"] = shape_handle
        self.tool_shapes["tool-head-1"] = shape_head

        head_position = [0.0, 0.12, 0.0]
        head_qua = [0,0,0,1]
        if self.cfg["secondary-rand"] == True:
            head_position, head_qua = self.add_head_randomness(head_position)

        print("head_position: ", head_position)

        self.C.addFrame("tool-handle-1", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-1"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setContact(1) \
            .setMass(0.05)
        self.C.addFrame("tool-head-1", "tool-handle-1") \
                .setShape(ry.ST.ssBox, self.tool_shapes["tool-head-1"]) \
                .setRelativePosition(head_position) \
                .setRelativeQuaternion(head_qua)\
                .setContact(1) \
                .setMass(0.05)
        
        self.grasp_choices['tool1'] = ["tool-handle-1", "tool-head-1"]
        self.test_tools.append("tool-handle-1")
        

    def create_plunger(self,loc):
        shape_handle = self.add_randomness([0.04, 0.25, 0.03, 0.005])
        shape_head = self.add_randomness([0.25, 0.04, 0.03, 0.005])

        self.tool_shapes["tool-handle-2"] = shape_handle
        self.tool_shapes["tool-head-2"] = shape_head

        head_position = [0.0, 0, 0.0]
        head_qua = hlp.quaternion_rotation_on_z_axis([0,0,0,1],180)
        if self.cfg["secondary-rand"] == True:
            head_position, head_qua = self.add_head_randomness(head_position, qua=10)


        self.C.addFrame("tool-handle-2", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-2"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setContact(1) \
            .setMass(0.05)

        self.C.addFrame("tool-head-2", "tool-handle-2") \
                .setShape(ry.ST.ssBox, self.tool_shapes["tool-head-2"]) \
                .setRelativePosition(head_position) \
                .setRelativeQuaternion(head_qua)\
                .setContact(1) \
                .setMass(.1)
        
        self.grasp_choices['tool2'] = ["tool-handle-2", "tool-head-2"]
        self.test_tools.append("tool-handle-2")
        

    def create_L_ruler(self, loc):
        shape_handle = self.add_randomness([0.04, 0.25, 0.03, 0.005])
        shape_head = self.add_randomness([0.12, 0.04, 0.03, 0.005])

        head_position = [0.06, -0.1, 0.0]
        head_qua = [0,0,0,1]
        if self.cfg["secondary-rand"] == True:
            head_position, head_qua = self.add_head_randomness(head_position)


        self.tool_shapes["tool-handle-3"] = shape_handle
        self.tool_shapes["tool-head-3"] = shape_head
        quaternion_handle = hlp.quaternion_rotation_on_z_axis([0,0,0,1],-180)
        
        self.C.addFrame("tool-handle-3", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-3"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setQuaternion(quaternion_handle)\
            .setContact(1) \
            .setMass(.01)
        self.C.addFrame("tool-head-3", "tool-handle-3") \
                .setShape(ry.ST.ssBox, self.tool_shapes["tool-head-3"]) \
                .setRelativePosition(head_position) \
                .setRelativeQuaternion(head_qua)\
                .setContact(1) \
                .setMass(.01)
        
        self.grasp_choices['tool3'] = ["tool-handle-3", "tool-head-3"]
        self.test_tools.append("tool-handle-3")
        
    
    def create_thick_stick(self, loc):

        shape_handle = self.add_randomness([0.05, 0.15, 0.05, 0.005])

        self.tool_shapes["tool-handle-4"] = shape_handle

        quaternion_handle = hlp.quaternion_rotation_on_z_axis([0,0,0,1],0)
        self.C.addFrame("tool-handle-4", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-4"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setQuaternion(quaternion_handle)\
            .setContact(1) \
            .setMass(.01)
        
        self.grasp_choices['tool4'] = ["tool-handle-4"]
        self.test_tools.append("tool-handle-4")
        

    def create_spatula(self, loc):
        shape_handle = self.add_randomness([0.02, 0.15, 0.02, 0.005])
        shape_head = self.add_randomness([0.08, 0.08, 0.01, 0.005])
    
        self.tool_shapes["tool-handle-5"] = shape_handle
        self.tool_shapes["tool-head-5"] = shape_head

        head_position = [0.0, 0.09, 0.0]
        head_qua = [0,0,0,1]
        if self.cfg["secondary-rand"] == True:
            head_position, head_qua = self.add_head_randomness(head_position)

        self.C.addFrame("tool-handle-5", "table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, self.tool_shapes["tool-handle-5"]) \
            .setRelativePosition(self.tool_loc[loc]) \
            .setContact(1) \
            .setMass(0.05)
        self.C.addFrame("tool-head-5", "tool-handle-5") \
                .setShape(ry.ST.ssBox, self.tool_shapes["tool-head-5"]) \
                .setRelativePosition(head_position) \
                .setRelativeQuaternion(head_qua)\
                .setContact(1) \
                .setMass(0.05)
        
        self.grasp_choices['tool5'] = ["tool-handle-5", "tool-head-5"]
        self.test_tools.append("tool-handle-5")
    
    def meaningless_1(self, loc):
        shape_handle = self.add_randomness([0.1, 0.1, 0.1, 0.005])
        shape_head = self.add_randomness([0.03, 0.03, 0.3, 0.005])
    
        self.tool_shapes["tool-handle-6"] = shape_handle
        self.tool_shapes["tool-head-6"] = shape_head


        rand0, rand1= 30,30
        if self.cfg["secondary-rand"] == True:
            rand0 = random.randint(-rand0,rand0)
            rand1 = random.randint(-rand1,rand1)

        self.C.addFrame("tool-handle-6","table") \
            .setJoint(ry.JT.rigid) \
            .setShape(ry.ST.ssBox, shape_handle) \
            .setRelativePosition([- 0.1, 0.20, 0.055]) \
            .setContact(1) \
            .setMass(0.05)
        self.C.addFrame("tool-head-6","tool-handle-6") \
            .setShape(ry.ST.ssBox, shape_head) \
            .setRelativePosition([0.0, 0.0, 0.1]) \
            .setContact(1) \
            .setMass(0.05)
        self.C.addFrame("tool-stick-2","tool-handle-6") \
            .setShape(ry.ST.ssBox, shape_head) \
            .setRelativePosition([0.0, 0.0, 0.1]) \
            .setQuaternion(hlp.quaternion_rotation([1,0,0,0],rand0,[0,1,0]))\
            .setContact(1) \
            .setMass(0.05)
        self.C.addFrame("tool-stick-3","tool-handle-6") \
            .setShape(ry.ST.ssBox, shape_head) \
            .setRelativePosition([0.0, 0.0, 0.1]) \
            .setQuaternion(hlp.quaternion_rotation([1,0,0,0],rand1,[1,0,0]))\
            .setContact(1) \
            .setMass(0.05)

        
        self.grasp_choices['tool6'] = ["tool-handle-6", "tool-head-6"]

    def update_probabilities(self, grasp_point, idx, success):

        prob = self.tool_attributes.orientation_probabilities[grasp_point]
        prob[idx] = prob[idx] + success*self.learning_rate - (1 -success)*self.learning_rate
        self.tool_attributes.orientation_probabilities[grasp_point] = prob/np.sum(prob)
        print("New Prob: ", self.tool_attributes.orientation_probabilities[grasp_point])

    def add_randomness(self, shape, change_factor = 0.1):
        random_shape = []
        for i in range(3):
            adjustment = shape[i] * change_factor
            adjusted_value = shape[i] + random.uniform(-1, 1) * adjustment
            random_shape.append(adjusted_value)
        
        random_shape.append(random.uniform(0.001,0.01))

        return random_shape
    
    def add_head_randomness(self, position, change = 0.008, qua = 20):
        random_translated = []
        
        random_translated.append(position[0] + random.uniform(-change,change))
        random_translated.append(position[1] + random.uniform(-change*2,change*2))
        random_translated.append(position[2] + random.uniform(-change,change))
        
        random_angle = random.randint(-qua,qua)
        randomized_qua = hlp.quaternion_rotation([0,0,0,1],random_angle,(0,0,1))

        return random_translated, randomized_qua


class ToolAttributes:
    def __init__(self):
        
        self.orientation_probabilities = {}
        self.initialize_orientation_probabilities()


    def initialize_orientation_probabilities(self):
        self.orientation_probabilities['tool-handle-0'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-handle-1'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-head-1'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-handle-2'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-head-2'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-handle-3'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-head-3'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-handle-4'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-handle-5'] = np.array([0.50, 0.50])
        self.orientation_probabilities['tool-head-5'] = np.array([0.50, 0.50])