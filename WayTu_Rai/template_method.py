import numpy as np
from scipy.spatial import KDTree
import os
import pickle

class TemplateMethod:
    def __init__(self, root):
        self.root = root
        self.sample_names = os.listdir(root)
        self.point_clouds = []
        for sample in self.sample_names:
            pc_path = os.path.join(root, sample, "point_cloud.npy")
            pc = np.load(pc_path)
            self.point_clouds.append(pc)

    def get_way_points(self, idx):
        
        path = os.path.join(self.root, self.sample_names[idx], "waypoints.pkl")
        with open(path, 'rb') as f:
            waypoints = pickle.load(f)
        return waypoints

    def find_nearest_neighbor(self, test):
        distances = []
        for i in range(len(self.point_clouds)):
            distances.append(self.chamfer_distance(self.point_clouds[i], test))
            print("Calculated: ", i)
        distances = np.array(distances)
        idx = np.argmin(distances)

        waypoints = self.get_way_points(idx)
        print("waypoints")
        return waypoints

    

    def chamfer_distance(self, train_pc, test_pc):
        train_tree = KDTree(train_pc)
        test_tree = KDTree(test_pc)

        distance_train = np.mean(train_tree.query(test_pc))
        distance_test = np.mean(test_tree.query(train_pc))

        return (distance_train + distance_test)/2
