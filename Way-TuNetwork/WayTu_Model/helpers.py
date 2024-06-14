import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import math
from itertools import permutations, combinations
from scipy.spatial.transform import Rotation
import torch

def get_all_samples(data_paths, val_size):
    sample_paths = []
    for folders in data_paths:
        for tool_class in os.listdir(folders):
            for samples in os.listdir(os.path.join(folders, tool_class)):
                sample_paths.append(os.path.join(folders,tool_class,samples))
    
    positive_sample_paths = []
    x_bound = []
    for sample  in sample_paths:
        with open(os.path.join(sample, 'waypoints.pkl'), 'rb') as f:
            infos = pickle.load(f)
            x_bound.append(infos[2][0])
            if infos[2][0] >= 0.3:
                positive_sample_paths.append(positive_sample_paths)  
        
    print("sample mean", np.mean(np.array(x_bound)))
    print(f"There are total {len(sample_paths)} samples in dataset. {len(positive_sample_paths)} is larger than 0.43")
    # data = np.array(sample_paths)
    data = np.array(sample_paths)
    np.random.shuffle(data)
    train_set, val_set= train_test_split(data,test_size=val_size, shuffle=True)
    print(f"There are total {data.shape[0]} samples in dataset.")
    upper_bound = get_score_upper_bound(sample_paths)
    return train_set, val_set, upper_bound

def normalize_sample(point_cloud, grasp_position, ready_position, go_position):
    center = np.mean(point_cloud, axis = 0)
    centered_point_cloud = point_cloud - center

    max_distance = np.max(np.linalg.norm(centered_point_cloud, axis=1))
    normalized_point_cloud = centered_point_cloud / max_distance

    normalized_grasp_position = (grasp_position - center)/max_distance
    normalized_ready_position = (ready_position - center)/max_distance
    normalized_go_position = (go_position - center)/max_distance


    return normalized_point_cloud, normalized_grasp_position, normalized_ready_position, normalized_go_position

def normalize_test_pc(point_cloud):
    center = np.mean(point_cloud, axis = 0)
    centered_point_cloud = point_cloud - center

    max_distance = np.max(np.linalg.norm(centered_point_cloud, axis=1))
    normalized_point_cloud = centered_point_cloud / max_distance

    return normalized_point_cloud, center, max_distance
def normalize_prediction(prediction, center, max_distance):
    prediction[:3] = prediction[:3] * max_distance + center
    prediction[3:7] = normalize_quaternions(prediction[3:7])

    prediction[7:10] = prediction[7:10] * max_distance + center
    prediction[10:14] = normalize_quaternions(prediction[10:14])

    prediction[14:17] = prediction[14:17] * max_distance + center
    prediction[17:21] = normalize_quaternions(prediction[17:21])
    return prediction

def get_score_upper_bound(data_paths):
    scores = []
    for data in data_paths:
        with open(os.path.join(data, 'waypoints.pkl'), 'rb') as f:
            infos = pickle.load(f)
            scores.append(infos[0])
    scores = np.asarray(scores, dtype=np.float32)
    return np.percentile(scores, 99)

def normalize_quaternions(quaternion):
        norm = np.linalg.norm(quaternion)
        return quaternion/norm

def save_failed(C, cfg, parameters):
    fail_folder_path = os.path.join(cfg["failed-path"], cfg['task'])
    num_fails = len(os.listdir(fail_folder_path))
    # Not finished

def get_combination(tool_list, num_tools):
    return combinations(tool_list, num_tools)

def euler_from_quaternion(quaternion):
        w,x,y,z = quaternion

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z

def euler_to_quaternion(waypoint_euler):
    print(waypoint_euler.shape)
    quaternions = []
    for i in range(3):
        rot = Rotation.from_euler('xyz', waypoint_euler[:,i,:], degrees=True)
        rot_quat = rot.as_quat()
        quaternions.append(rot_quat)
    
    quaternions = torch.from_numpy(np.asarray(quaternions, dtype=np.float32))
    quaternions = quaternions.permute(1, 0, 2)
    return quaternions
   

class TestStatistics:
    def __init__(self, cfg):
        self.cfg = cfg
        base = [- 0.1, 0.20, 0.055]
        distance_x = 0.23

        self.tool_loc = [[base[0], base[1], 0.055],
                                   [base[0] - distance_x, base[1], 0.057],
                                   [base[0] + distance_x, base[1], 0.057],]
        self.grasp_choices = {}
        self.add_grasp_choices()

        self.statistics = {}  
        for tool in list(self.grasp_choices.keys()):
            self.statistics[tool] ={"placed"    :0, 
                                    "selected"  :0,
                                    "half"      :0, 
                                    "success"   :0, 
                                    "score"     :[],
                                    "placement" :[]}
            
        self.continue_statistics = {
            "t1": 0,
            "t2": 0,
            "t3" : 0,
            "cannot": 0,
        }
        self.loc_statistics = [0,0,0]
    
    def add_grasp_choices(self):
        self.grasp_choices['tool0'] = ["tool-handle-0"]
        self.grasp_choices['tool1'] = ["tool-handle-1", "tool-head-1"]
        self.grasp_choices['tool2'] = ["tool-handle-2", "tool-head-2"]
        self.grasp_choices['tool3'] = ["tool-handle-3", "tool-head-3"]
        self.grasp_choices['tool4'] = ["tool-handle-4"]
        self.grasp_choices['tool5'] = ["tool-handle-5", "tool-head-5"]
        self.grasp_choices['tool6'] = ["tool-handle-6", "tool-head-6"]
    
    def update_selected(self, tool_dict, tool_locations, C):
        count = 0
        for tool in tool_dict:
            name = 'tool' + tool[-1]
            print("name: ", name)
            self.statistics[name]["placed"] = self.statistics[name]["placed"] + 1
            self.statistics[name]["placement"].append(C.getFrame(tool).getPosition())
            count +=1
        return tool_dict
    

    def find_tool(self, selected_tools, grasp_waypoint):
        print("grasp_Waypoint:", grasp_waypoint)
        distances = []
        for tool in selected_tools:
            print("tool: ", tool)
            tool = 'tool' + tool[-1]
            dist = math.sqrt((self.statistics[tool]["placement"][-1][0] - grasp_waypoint[0]) ** 2 +
                              (self.statistics[tool]["placement"][-1][1] - grasp_waypoint[1]) ** 2+ 
                              (self.statistics[tool]["placement"][-1][2] - grasp_waypoint[2]) ** 2)
            distances.append(dist)
            print("dist: ", dist)
            print(self.statistics[tool]["placement"][-1])
        print(selected_tools)
        index_min = np.argmin(np.array(distances))
        selected = selected_tools[index_min]
        print(selected_tools)
        print(selected)

        self.loc_statistics[index_min] += 1 

        return 'tool' + selected[-1]
    def update_waypoint(self, selected, score):

        self.statistics[selected]["selected"] = self.statistics[selected]["selected"] + 1

        if self.cfg['task'] == 'reach':
            threshold,success = 0.41, 0.43
        elif self.cfg['task'] == 'minigolf':
            threshold, success  = 0.4, 0.6
        elif self.cfg['task'] == 'lift':
            threshold,success = 0.41, 0.43

        if score > success : # 0.015:
            self.statistics[selected]["success"] = self.statistics[selected]["success"] + 1
            self.statistics[selected]["score"].append(score)
        elif score > threshold: 
            self.statistics[selected]["half"] = self.statistics[selected]["half"] + 1
            self.statistics[selected]["score"].append(score)


    
    def update_continue_statistics(self, trial_count,solve):
        if solve == True:
            key_name = 't' + str(trial_count+1)
            self.continue_statistics[key_name] = self.continue_statistics[key_name] + 1
        else: 
            self.continue_statistics['cannot'] =self.continue_statistics['cannot'] + 1 
    
    def print_statistics(self):
        for tool, stats in self.statistics.items():
            print(f"Tool: {tool}")
            print(f"  Placed: {stats['placed']}")
            print(f"  Selected: {stats['selected']}")
            print(f"  Success: {stats['success']}")
            print(f"  Half: {stats['half']}")

            
            if stats['score']:
                average_score = sum(stats['score']) / len(stats['score'])
            else:
                average_score = 0
            
            print(f"  Average Score: {average_score:.2f}")
            print(self.loc_statistics)
            print()
    def print_continue_statistics(self):
        print(self.continue_statistics)
    
        
