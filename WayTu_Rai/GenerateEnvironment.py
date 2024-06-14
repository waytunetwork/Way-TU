import robotic as ry
import numpy as np
import time
import random

from WayTu_Rai.helpers_with_komo import PushTask
from WayTu_Rai.ReachEnvironment import ReachTask
from WayTu_Rai.MiniGolfEnvironment import MiniGolfTask
from WayTu_Rai.LiftEnvironment import LiftTask
from WayTu_Rai.template_method import TemplateMethod
from WayTu_Rai.GenerateTools import CreateTools
import WayTu_Rai.helper_functions as hlp

class GenerateEnvironment:
    def __init__(self, C, task, cfg, tool_att = None, obj_waypoint_att = None, test_sta = None) -> None:
        # This shows where is the hammer is positioned. The objects will be positioned opposite of the tool
        self.side = np.random.choice(a=[-1,1])
        self.task = task
        self.C = C
        self.cfg = cfg

        self.tool_att = tool_att
        self.obj_waypoint_att = obj_waypoint_att
        self.test_sta = test_sta
    
    def create_environment(self, num_tools = 6, test_list = None):
        """
        For creating environment
        Input: 
            tool --> Every tool has different head, tool parameters decides which head will be attached to the tool 
        """
        self.C.addFile('/home/.../git/RAI/Keto/panda_camera.g')
        self.C.addFile('rai-robotModels/scenarios/pandaSingle.g')

        self.tool_obj = CreateTools(self.C, self.tool_att,self.cfg)

        if test_list == None:
            self.create_num_tool_random(num_tools)
        else: 
            self.create_test_tools(test_list)

        self.env = None

        parameters = None  
        if self.task == "reach":
            self.env = ReachTask(self.C, self.side, self.obj_waypoint_att)
            self.env.create_new_reach_static()
        elif self.task == "minigolf":
            self.env = MiniGolfTask(self.C, self.obj_waypoint_att)
            self.env.create_minigolf_static()
        elif self.task == "lift":
            self.env = LiftTask(self.C, self.obj_waypoint_att)
            self.env.create_lift_static()
    
    def set_tool_handle(self, tool_handle):
        self.tool_handle = tool_handle
        self.tool_start_pos = self.C.getFrame(self.tool_handle).getPosition()
    
    def get_grasp_Score(self):
        new_position = self.C.getFrame(self.tool_handle).getPosition()
        obj_new = self.C.getFrame("obj").getPosition()
        return self.env.grasp_score(self.tool_start_pos,new_position, obj_new)
    
    def set_tool_pc(self, tool_pc):
        self.tool_pc = tool_pc
    
    def create_test_tools(self, num_tool_list):

        tool_list = self.tool_obj.tool_list[num_tool_list]
        tool_locations = self.tool_obj.tool_loc[:len(tool_list)]
        print(tool_locations)
        random_idx = random.sample(range(0,3), 3)
        print(tool_locations)
        for i in range(len(tool_list)):
            tool_list[random_idx[i]](i)

        if self.test_sta is not None:
            selected_tools = self.test_sta.update_selected(self.tool_obj.grasp_choices)
            self.selected_tools = selected_tools

        self.grasp_choices = self.tool_obj.grasp_choices
        self.base = self.tool_obj.base
        self.distance = self.tool_obj.distance
        

    def create_num_tool_random(self,num_tools):
        
        # Get locations and suffle them for randomness
        tool_locations = self.tool_obj.tool_loc[:num_tools]
        random.shuffle(tool_locations)

        tool_func = self.tool_obj.create_functions
        random.shuffle(tool_func)
        print(len(tool_func))
        print(num_tools)

        for i in range(num_tools):
            tool_func[i](i)
        
        if self.test_sta is not None:
            selected_tools = self.test_sta.update_selected(self.tool_obj.test_tools, tool_locations, self.C)
            self.selected_tools = selected_tools

        self.grasp_choices = self.tool_obj.grasp_choices
        self.base = self.tool_obj.base
        self.distance = self.tool_obj.distance
    
    def update_grasp_orientation_probabilities(self, tool_choice, idx, success):
        self.tool_obj.update_probabilities(tool_choice, idx, success)

    def update_obj_orientation_probabilities(self, tool_choice, idx, success):
        self.env.update_probabilities(tool_choice, idx, success)


    def create_handle_waypoint(self, waypoint = None, heuristic = False):
        if waypoint is None: 
            if heuristic == True: 
                greedy = np.random.uniform(0,1)
                if greedy > 0.5:  
                    prob_dist = self.tool_obj.tool_attributes.orientation_probabilities[self.tool_handle]
                    prob_idx  = random.choices(range(len(self.tool_obj.orientations)), k=1, weights=prob_dist)[0]
                    orientation = self.tool_obj.orientations[prob_idx]

                    print(self.tool_handle + ": ")
                    print("probability"+ str(self.tool_obj.tool_attributes.orientation_probabilities[self.tool_handle]))

                else:
                    orientation = self.C.getFrame(self.tool_handle).getQuaternion()
                    prob_idx = -1

                handle_shape =  self.tool_obj.tool_shapes[self.tool_handle]
                print("handle_shape: ", handle_shape)
                max_length = np.max(np.array(handle_shape))
                randomize = np.random.uniform(-max_length/2.1 , max_length/2.1)
                position = self.C.getFrame(self.tool_handle).getPosition() + [0.0, randomize, -0.01]
        
            else:
                handle_shape = self.hammer_parameters["handle_shape"]
                
                orientation = self.C.getFrame(self.tool_handle).getQuaternion()
                p = 0.5  
                if p< 0.3: 
                    position = self.random_grasp()
                elif p >= 0.3 and p<0.6: 
                    position = self.env.grasp_static()
                elif p>= 0.6:
                    position = self.random_grasp_point()
                # position = self.random_grasp_point(self.tool_pc)
                # position = self.generate_grasp_way_point_static()
            # if self.task == "minigolf":
            #     orientation = hlp.quaternion_rotation(orientation,30,[1,0,0])
            self.prob_idx = prob_idx
        else:
            position = waypoint[:3]
            orientation =  waypoint[3:7]

        self.C.addFrame('waypoint-tool')\
                .setShape(ry.ST.marker, size=[.1])\
                .setQuaternion(orientation)\
                .setPosition(position)
        
    def update_cross(self, selected):
        print("selected is: ",selected)
        if selected == 'tool2':
            correction = self.C.getFrame('waypoint-tool').getPosition()
            correction[1] += 0.08 
            self.C.getFrame('waypoint-tool').setPosition(correction)

    def random_grasp_point(self):
        pc =  np.asarray(self.tool_pc)
        pc_size = pc.shape[0]
        random_point_idx = np.random.choice(pc_size)
        random_point = self.tool_pc[random_point_idx]
        z_coords = pc[:, 2]
        min_z, max_z = min(z_coords), max(z_coords)

        random_z = np.random.uniform(min_z, max_z)
        return random_point
    
    def random_grasp(self):
        # Create a bounding box around the point cloud 
        min_corner = np.min(self.tool_pc, axis=0)
        max_corner = np.max(self.tool_pc, axis=0)

        midpoints = (min_corner + max_corner) / 2
        std_dev = (max_corner - min_corner) / 6

        random_point = np.random.normal(loc=midpoints, scale=std_dev, size=(1, len(min_corner)))
        random_point = np.clip(random_point, min_corner, max_corner)  # Ensure points are within bounds

        return random_point
        
    def create_waypoints(self, waypoint = None):
        # Create Handle Waypoint
        self.create_handle_waypoint(waypoint=waypoint,heuristic=True)

        # Create Object Waypoints
        if self.task == "push":
            ball1 = self.C.getFrame('ball-left').getPosition()
            ball2 = self.C.getFrame('ball-middle').getPosition()
            ball3 = self.C.getFrame('ball-right').getPosition()

            ready_pose, go_pose = self.env.find_object_waypoint(ball1, ball2, ball3)
            object_rotation = self.object_rotation 
            
        elif self.task == "reach":
            # print("Here")
            if waypoint is None: 
                ready_pose, go_pose = self.env.find_reach_object_waypoints_static()
                object_rotation = self.C.getFrame("obj").getQuaternion()
                object_rotation = hlp.quaternion_rotation_on_z_axis(object_rotation, 90)
                if self.cfg['train']==True:
                    obj_rot_idx  = random.choices(range(len(self.tool_obj.obj)), k=1)[0]
                    object_rotation = self.tool_obj.obj[obj_rot_idx]
                ready_rotation = object_rotation
                go_rotation = object_rotation
            else: 
                ready_pose = waypoint[7:10]
                go_pose = waypoint[14:17]

                ready_rotation = waypoint[10:14]
                go_rotation = waypoint[17:21]
                

        elif self.task == "minigolf":
            if waypoint is None:
                ready_pose, go_pose = self.env.find_minigolf_waypoints_static(tool_shape=self.tool_obj.tool_shapes[self.tool_handle])
                object_rotation = self.C.getFrame("obj").getQuaternion()
                if self.cfg['train']==True:
                    obj_rot_idx  = random.choices(range(len(self.env.obj)), k=1)[0]
                    object_rotation = self.env.obj[obj_rot_idx]
                    random_rotation = random.uniform(-60, +60)
                    object_rotation = hlp.quaternion_rotation(object_rotation,random_rotation,[1,0,0])

                ready_rotation = object_rotation
                go_rotation = object_rotation
            else: 
                ready_pose = waypoint[7:10]
                go_pose = waypoint[14:17]

                ready_rotation = waypoint[10:14]
                go_rotation = waypoint[17:21]

        elif self.task == "lift":
            if waypoint is None:
                ready_pose, go_pose = self.env.find_lift_waypoints_static()
                object_rotation = self.C.getFrame("obj").getQuaternion()
                if self.cfg['train']==True:
                        obj_rot_idx  = random.choices(range(len(self.tool_obj.obj)), k=1)[0]
                        object_rotation = self.tool_obj.obj[obj_rot_idx]
                        random_rotation = random.uniform(-30, +30)
                        object_rotation = hlp.quaternion_rotation(object_rotation,random_rotation,[0,0,1])
                ready_rotation = object_rotation
                go_rotation = object_rotation
            else: 
                ready_pose = waypoint[7:10]
                go_pose = waypoint[14:17]

                ready_rotation = waypoint[10:14]
                go_rotation = waypoint[17:21]

        else:
            raise Exception('The task is not in task list')

        way_ready = self.C.addFrame('waypoint-ready')
        way_go = self.C.addFrame('waypoint-go')

        way_ready.setShape(ry.ST.marker, size=[.1])
        way_go.setShape(ry.ST.marker, size=[.1])

        way_ready.setPosition(ready_pose)
        way_ready.setQuaternion(ready_rotation)
        
        way_go.setPosition(go_pose)
        way_go.setQuaternion(go_rotation)

        direction = [go_pose[0] - ready_pose[0], go_pose[1] - ready_pose[1], go_pose[2] - ready_pose[2]]

        middle1 = (ready_pose[0] + direction[0] / 3, ready_pose[1] + direction[1] / 3, ready_pose[2] + direction[2] / 3)
        middle2 = (ready_pose[0] + 2 * direction[0] / 3, ready_pose[1] + 2 * direction[1] / 3, ready_pose[2] + 2 * direction[2] / 3)

        way_middle_1 = self.C.addFrame('waypoint-middle-1')
        way_middle_1.setPosition(middle1)
        way_middle_1.setQuaternion(ready_rotation)
        way_middle_2 = self.C.addFrame('waypoint-middle-2')
        way_middle_2.setPosition(middle2)
        way_middle_2.setQuaternion(go_rotation)


        if self.task == 'reach' or self.task == 'lift':
            way_before_ready = self.C.addFrame('waypoint-before-ready')

            tool_pose = self.C.getFrame('waypoint-tool').getPosition()

            way_before_ready_pos = [tool_pose[0], tool_pose[1], ready_pose[2]-(ready_pose[2]-tool_pose[2])/2]
            way_before_ready.setPosition(way_before_ready_pos)
            way_before_ready.setQuaternion(ready_rotation)

    
    def template_waypoints(self, pcl):
        push_samples_path = "/home/.../git/Way-Tu/samples"
        template = TemplateMethod(push_samples_path)
        waypoints = template.find_nearest_neighbor(pcl.point_cloud.points)