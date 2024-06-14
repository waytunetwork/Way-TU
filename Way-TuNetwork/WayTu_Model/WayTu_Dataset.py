import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np
import WayTu_Model.helpers as hlp 

class WayTuDataset(Dataset):
    def __init__(self, data_paths, task, upper_bound = np.inf, normalize = True, data_augmentation = True):
        self.sample_paths = data_paths

        self.normalize = normalize
        self.data_augmentation = data_augmentation

        self.upper_bound = upper_bound
        self.task= task
        self.euler =False
    
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        # Get path of sample folder 
        path = self.sample_paths[idx]

        # Get score and waypoint information about sample
        with open(os.path.join(path, 'waypoints.pkl'), 'rb') as f:
            infos = pickle.load(f)
        
        # Get point cloud 
        point_cloud = np.load(os.path.join(path, "tool_point_cloud.npy"))
        # Split info into variables
        grasp_position, ready_position, go_position = infos[0], infos[2], infos[4] 
        grasp_quaternion, ready_quaternion, go_quaternion = infos[1], infos[3], infos[5]

        # Data Augmentation
        if self.data_augmentation == True:
            point_cloud, grasp_position = self.add_random_augmentation(point_cloud, grasp_position)

        # Normalize
        if self.normalize == True:
            point_cloud, grasp_position,ready_position, go_position = hlp.normalize_sample(point_cloud, grasp_position,ready_position, go_position)
            grasp_quaternion = hlp.normalize_quaternions(grasp_quaternion)
            ready_quaternion = hlp.normalize_quaternions(ready_quaternion)
            go_quaternion = hlp.normalize_quaternions(go_quaternion)
        
        # Concatenate waypoint variables as a vector 
        waypoint = np.concatenate((grasp_position, grasp_quaternion, ready_position, ready_quaternion, go_position, go_quaternion), axis=0)

        # Score function operations
        score= infos[6]
        if self.task == 'reach' or self.task == 'lift':
                score = self.update_score(infos[6])
        if self.euler == True:
            grasp_euler = hlp.euler_from_quaternion(grasp_quaternion)
            ready_euler = hlp.euler_from_quaternion(ready_quaternion) 
            go_euler = hlp.euler_from_quaternion(go_quaternion) 

            waypoint = np.concatenate((grasp_position, grasp_euler, ready_position, ready_euler, go_position, go_euler), axis=0)


        return point_cloud, waypoint, score
    
    def add_noise(self, point_cloud, noise_level = 0.01):
        noise = np.random.normal(scale=noise_level, size=point_cloud.shape)
        return point_cloud + noise
    
    def add_translation(self, point_cloud, waypoint):
        translation_vector = np.random.uniform(-0.07, 0.07, size=2)
        translation_vector = np.append(translation_vector, 0)

        point_cloud = point_cloud + translation_vector
        waypoint[:2] = waypoint[:2] + translation_vector[:2]

        return point_cloud, waypoint

    def add_random_augmentation(self, point_cloud, waypoint):
        if  np.random.rand() < 0.5:
            point_cloud, waypoint = self.add_translation(point_cloud, waypoint)
        if  np.random.rand() < 0.5:
            point_cloud = self.add_noise(point_cloud)

        
        return point_cloud, waypoint
    
    def update_score(self, score):
        score = np.clip(score, a_min=None, a_max=self.upper_bound)
        new_score = (score - 0.4)/0.6
        return new_score
        