import robotic as ry
import numpy as np
import time
# import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from WayTu_Rai.template_method import TemplateMethod
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class PointCloud:
    def __init__(self, C, cam_names, table_threshold, base, distance):
        self.cam_names = cam_names
        self.cam_list = []

        self.num_cam = len(cam_names)

        self.table_threshold = table_threshold
        # self.tool_handle = tool_handle

        self.C = C
        base[2] = base[2] + 0.6
        self.base = base
        self.distance = distance
        
        self.setCameras()
    
    def setCameras(self):
        for cam in self.cam_names:
            cam_temp = ry.CameraView(self.C)
            cam_temp.setCamera(cam)
            self.cam_list.append(cam_temp)
    
    def getPointCloud(self, only_tool = False):

        all_point_clouds = -1
        
        for i in range(self.num_cam):
            # Get Camera from camera list
            cam = self.cam_list[i]
            # Compute the rgb, depth information of the view of the camera
            _, depth = cam.computeImageAndDepth(self.C)
            # Calculate Point Cloud 
            point_cloud = ry.depthImage2PointCloud(depth, cam.getFxyCxy())
            # Get Camera Frame to translate objects to world frame
            cam_frame = self.C.getFrame(self.cam_names[i])
            # Get rid of third dimension
            point_cloud = point_cloud.reshape((point_cloud.shape[0] * point_cloud.shape[1], point_cloud.shape[2]))
            # Translate to world frame
            point_cloud = self.cam_to_world(point_cloud, cam_frame)
            # Remove Table
            point_cloud =  point_cloud[point_cloud[:, 2] > self.table_threshold]

            # Concatenate point clouds
            if i == 0 :
                all_point_clouds = point_cloud
            else:  
                all_point_clouds = np.concatenate((all_point_clouds, point_cloud), axis=0)
        if only_tool == True:
            # point_cloud = self.get_only_tool(point_cloud)
            all_point_clouds = self.get_only_5_tools(all_point_clouds)
        
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points =  o3d.utility.Vector3dVector(all_point_clouds)


    def calculate_density_based_voxel_size(point_cloud_np):
        # Use NearestNeighbors to find the distance to the nearest point
        nbrs = NearestNeighbors(n_neighbors=2).fit(point_cloud_np)
        distances, _ = nbrs.kneighbors(point_cloud_np)
        
        # The second column of distances contains the distances to the nearest neighbor (excluding self)
        average_distance = np.mean(distances[:, 1])
        
        # Set the voxel size to some multiple of the average nearest neighbor distance
        voxel_size = average_distance * 2  # Adjust the multiplier based on desired density
        return voxel_size
    
    def upsample (self, num_points, pc = None):
        if pc is not None: 
            points = pc
        else: 
            points = np.asarray(self.point_cloud.points)
            print("after: ", points.shape[0])

        missing = num_points - points.shape[0]

        if missing >= points.shape[0]:
            while True:
                if missing <= points.shape[0]:
                    break
                points = np.repeat(points, 2, axis = 0)
                missing = num_points - points.shape[0]
        if missing <= 0: 
            random_indices = np.random.choice(points.shape[0], size=num_points, replace=False)
            selected_rows = points[random_indices]
            return selected_rows
        else:
            random_indices = np.random.choice(points.shape[0], size=missing, replace=False)
            selected_rows = points[random_indices]
            return np.vstack((points, selected_rows))


    
    def get_only_tool(self, point_cloud):
        min_x , max_x = self.base[0] - 2*self.distance, self.base[0] + 2*self.distance
        min_y , max_y = self.base[1] - 0.5*self.distance, self.base[1] + 1.5*self.distance
        min_z , max_z = self.base[2] - 0.05, self.base[2] + 0.05  


        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1]
        z_coords = point_cloud[:, 2]

        x_within_bounds = (x_coords >= min_x) & (x_coords <= max_x)
        y_within_bounds = (y_coords >= min_y) & (y_coords <= max_y)
        z_within_bounds = (z_coords >= min_z) & (z_coords <= max_z)

        within_bounds = x_within_bounds & y_within_bounds & z_within_bounds

        tool_points = point_cloud[within_bounds]

        return tool_points
    
    def get_only_5_tools(self, point_cloud):
 

        min_x , max_x = self.base[0] - 1.4*self.distance, self.base[0] + 1.4*self.distance
        min_y , max_y = self.base[1] - 0.7*self.distance, self.base[1] + 1.9*self.distance
        min_z , max_z = self.base[2] - 0.05, self.base[2] + 0.05   

        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1]
        z_coords = point_cloud[:, 2]

        x_within_bounds = (x_coords >= min_x) & (x_coords <= max_x)
        y_within_bounds = (y_coords >= min_y) & (y_coords <= max_y)
        z_within_bounds = (z_coords >= min_z) & (z_coords <= max_z)

        within_bounds = x_within_bounds & y_within_bounds & z_within_bounds

        tool_points = point_cloud[within_bounds]

        return tool_points


    def segmentation(self):
        self.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
        clustering = DBSCAN(eps=0.05, min_samples=50, algorithm='brute',n_jobs=-1).fit(np.array(self.point_cloud.points))

        self.labels = clustering.labels_

        self.max_label = self.labels.max() 
        
        pc_array = self.get_different_tools()

        self.pc_array = np.array(pc_array)

    def find_selected_tool(self):
        grasp_pos = self.C.getFrame("waypoint-tool").getPosition()
        grasp_pos = np.array(grasp_pos)
        tool_means = []
        for i in range(self.pc_array.shape[0]):
            avg = np.mean(self.pc_array[i], axis=0)
            tool_means.append(avg)
        tool_means= np.array(tool_means)

        distances = np.linalg.norm(tool_means - grasp_pos, axis =1)
        tool_idx = np.argmin(distances)

        return self.pc_array[tool_idx]
        


    def get_different_tools(self):
        pc_array = []
        for j in range(self.max_label+1):

            indices = np.where(self.labels == j)[0]
            tmp_pc = self.point_cloud.select_by_index(indices)
            tmp_pc_arr = np.array(tmp_pc.points)

            tmp_pc_arr = self.upsample(2000, tmp_pc_arr)
            

            pc_array.append(tmp_pc_arr)
        return pc_array 

    def discard_robot(self):
        distances = []
        pc_array = []
        for j in range(self.labels.max()+1):
            indices = np.where(self.labels == j)[0]
            tmp_pc = self.point_cloud.select_by_index(indices)
            tmp_pc_arr = np.array(tmp_pc.points)

            max_value, min_value = np.max(tmp_pc_arr[:, -1]), np.min(tmp_pc_arr[:, -1])
            distances.append(max_value - min_value)
            pc_array.append(tmp_pc_arr) 
        
        index_of_largest_distance = np.argmax(distances)
        pc_array.pop(index_of_largest_distance)
        
        return pc_array


    def cam_to_world(self, point_cloud, cam_frame):
        t = cam_frame.getPosition() 
        R = cam_frame.getRotationMatrix()
        points_camera_frame = point_cloud

        # Add homogeneous coordinates for the points in camera frame
        points_camera_frame_homogeneous = np.hstack((points_camera_frame, np.ones((points_camera_frame.shape[0], 1))))
        # Transformation matrix (combining rotation and translation)
        transformation_matrix = np.vstack((np.hstack((R, t.reshape(-1, 1))), np.array([0, 0, 0, 1])))
        # Transform all points to world frame
        points_world_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T
        # Extract the 3D coordinates in the world frame
        points_world_frame = points_world_frame_homogeneous[:, :3]
        return points_world_frame
    
    def visualize_point_clouds(self):
        o3d.visualization.draw_geometries([self.point_cloud])
    
    def visualize_pc_from_array(self, pc):
        pcl = o3d.geometry.PointCloud()
        pcl.points =  o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcl])