
import yaml
import robotic as ry
import numpy as np
import time
import torch
import WayTu_Rai.helpers_with_komo as hlp
from WayTu_Rai.GenerateTools import ToolAttributes, CreateTools
from WayTu_Rai.PointCloud import PointCloud
from WayTu_Rai.GenerateEnvironment import GenerateEnvironment
from WayTu_Model.WaytuModel import GraspScoreModel,WaypointGeneratorModel, WayTuModel
import WayTu_Model.test_model as model
from WayTu_Rai.ReachEnvironment import ReachWaypointAttributes
from WayTu_Rai.MiniGolfEnvironment import MiniGolfWaypointAttributes
from WayTu_Rai.LiftEnvironment import LiftWaypointAttributes
import WayTu_Model.helpers as helper
from WayTu_Model.helpers import TestStatistics
import random

import os
import pickle
import multiprocessing


def get_config():
    with open('config.yaml', 'r') as f:
        # data = yaml.load(f, Loader=yaml.SafeLoader)
        data = yaml.safe_load(f)
    print(data)
    return data

def see_generated_waypoints(C, waypoint_proposals):
    # 'waypoint-tool'
    tool = 0
    for waypoint in waypoint_proposals:
        position = waypoint[:3]
        orientation = waypoint[3:7]
        name = 'waypoint-tool-' + str(tool)
        add_waypoint(C, position, orientation, name)
        tool += 1
        print(position, " ", orientation)
    C.view()
    time.sleep(1)
def add_waypoint(C, position, orientation, name):
    C.addFrame(name)\
        .setShape(ry.ST.marker, size=[0.3])\
        .setQuaternion(orientation)\
        .setPosition(position)

def real_end_to_end_test(cfg):
    test_sta = TestStatistics(cfg)

    for i in range(cfg['num-trials']):
        C = ry.Config()
        env = GenerateEnvironment(C, cfg['task'], cfg, test_sta=test_sta)

        env.create_environment(num_tools=cfg["num-tools"])

        camera_names = ["cameraTop1", "cameraTop2", "cameraTop3"]
        pcl = PointCloud(C, camera_names, 0.659, env.base, env.distance)
        pcl.getPointCloud(only_tool=True)
        # pcl.visualize_point_clouds()
        pcl.segmentation()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = WayTuModel(cfg['feature-extractor'], num_points= cfg['num-points'],num_waypoints=3)
        model.load_state_dict(torch.load(cfg['e2e-model']))
        model.eval()
        model.to(device)

        scores = []
        waypoint_proposals = []
        for i in range(cfg['num-tools']):
            pc = pcl.pc_array[i]
            if cfg['e2e-normalize'] == True:
                pc, center, max_distance = helper.normalize_test_pc(pc)
            pc = torch.from_numpy(pc)
            pc = pc.unsqueeze(0)

            waypoint_positions, waypoint_quaternion, score = model(pc.to(device))

            waypoint = torch.cat((waypoint_positions, waypoint_quaternion), dim=-1).view(waypoint_positions.size(0), -1)

            if cfg['w-normalize'] == True:
                waypoint = waypoint.cpu().detach().numpy().flatten()
                waypoint = helper.normalize_prediction(waypoint, center, max_distance)
            
            scores.append(score.item())
            waypoint_proposals.append(waypoint)

        for i in range(cfg['num-tools']):
            print(f"waypoint: {waypoint_proposals[i][:3]}, score: {scores[i]}")

        print("I am here")
        if cfg["continue-solve"] == False:
            best_waypoint_idx = np.argmax(np.array(scores))
            best_waypoint = waypoint_proposals[best_waypoint_idx]
            env.create_waypoints(waypoint=best_waypoint)
            selected = test_sta.find_tool(env.selected_tools, best_waypoint[:3])
            env.set_tool_handle(test_sta.grasp_choices[selected][0])
            env.update_cross(selected)
            pose_parameters, dance = hlp.tool_manipulation(C, 'Heuristic', cfg['task'], env.selected_tools, env.tool_obj.tool_loc[:cfg['num-tools']])

            score, grasp_success = env.get_grasp_Score()
            test_sta.update_waypoint(selected,score)
            print(f"selected: {selected} score: {score}")

            print("ready, quaternion:" , best_waypoint[10:14])
        else:
            combined = list(zip(waypoint_proposals, scores))
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True) 

            sorted_waypoint_proposals, sorted_scores = zip(*sorted_combined)
            sorted_waypoint_proposals = list(sorted_waypoint_proposals)
            sorted_scores = list(sorted_scores)

            print("here")

            solve = False
            for i in range(cfg["num-tools"]):
                env.create_waypoints(waypoint=sorted_waypoint_proposals[i])
                selected = test_sta.find_tool(env.selected_tools, sorted_waypoint_proposals[i][:3])
                env.set_tool_handle(test_sta.grasp_choices[selected][0])
                env.update_cross(selected)

                _, _ = hlp.tool_manipulation(C, 'Heuristic', cfg['task'])

                
                score, grasp_success = env.get_grasp_Score()
                print("Score: ", score)

                if cfg['task'] == 'reach':
                    threshold,success = 0.42, 0.43
                elif cfg['task'] == 'minigolf':
                    threshold, success  = 0.4, 0.6
                elif cfg['task'] == 'lift':
                    threshold,success = 0.42, 0.43
                    
                if score > threshold :
                    solve = True
                    break
            test_sta.update_continue_statistics(trial_count=i, solve=solve)
        test_sta.print_statistics()
        test_sta.print_continue_statistics()

        

def collect_sample(cfg):
    # Hyperparameters: 
    task = cfg['task'] 
    path_method = "Heuristic"
    save_sample = cfg['save-sample']
    save_environment = False
    test = False 
    path_folder = cfg['dataset-path']
    num_tool_choice = cfg['num-tools']
    start, end = 0, cfg['num-trials']
    count = 0



    # Created main folder for samples
    if not os.path.exists("./" + path_folder):
        os.mkdir("./" + path_folder)

    tool_idx = []

    # Create sub-folders for different tools
    for j in range(num_tool_choice):
        tool_folder_path = os.path.join("./", path_folder,"tool_" + str(j))
        if not os.path.exists(tool_folder_path):
            os.mkdir(tool_folder_path)
            tool_idx.append(0)
        else:
            samples = os.listdir(tool_folder_path)
            tool_idx.append(len(samples))


    # Generate Tool Attributes that will use in all experiments
    tool_att = ToolAttributes()
    if cfg["task"] == 'reach':
        obj_waypoint_att = ReachWaypointAttributes()
    elif cfg["task"] == 'minigolf':
        obj_waypoint_att = MiniGolfWaypointAttributes()
    elif cfg["task"] == 'lift':
        obj_waypoint_att = LiftWaypointAttributes()
    else: 
        raise Exception("The task is wrong. There is no environment for given task.")
        
    start_time = time.time()
    count, success_count, triple_success = 0, 0,0  
    for i in range(start,end):
        C = ry.Config()
        print(tool_att.orientation_probabilities)
        env = GenerateEnvironment(C,task, cfg, tool_att, obj_waypoint_att)

        if test:
            new_path_env = "./test/environment_" + str(i) + "/" 
        else:
            num_tools = cfg['num-tools']
            env.create_environment(num_tools)

            tool_list = list(env.grasp_choices.keys())
            selected_tool = random.choice(tool_list)
            tool_choice = random.choice(env.grasp_choices[selected_tool])
            tool_choice_idx = tool_list.index(selected_tool)

            # A setter function that set where to grasp the object. 
            env.set_tool_handle(tool_choice)
    
        # # ---------------------------------------------
        camera_names = ["cameraTop1", "cameraTop2", "cameraTop3"]
        pcl = PointCloud(C, camera_names, 0.659, env.base, env.distance)
        pcl.getPointCloud(only_tool=True)
        # pcl.visualize_point_clouds()
        pcl.segmentation()
        env.set_tool_pc(pcl.point_cloud.points)

        # Below is for template method
        if path_method == "Template": 
            env.template_waypoints(pcl)
        else: 
            env.create_waypoints()
        tool_pc = pcl.find_selected_tool()
        # pcl.visualize_pc_from_array(tool_pc)
        
        way_handle = C.getFrame("waypoint-tool")
        way_go =  C.getFrame("waypoint-go")
        way_ready = C.getFrame("waypoint-ready")

        # Save waypoint positions
        way_handle_position, way_handle_qua = way_handle.getPosition(),  way_handle.getQuaternion()
        way_go_position, way_go_qua = way_go.getPosition(),  way_go.getQuaternion()
        way_ready_position, way_ready_qua = way_ready.getPosition(),  way_ready.getQuaternion()
        
        pose_parameters, dance = hlp.tool_manipulation(C, path_method, task)
        
        success = False
        
        count +=1
        score, grasp_success = env.get_grasp_Score()
        if env.prob_idx > -1:
            env.update_grasp_orientation_probabilities(tool_choice, env.prob_idx, grasp_success)
            env.update_obj_orientation_probabilities(tool_choice, env.prob_idx, grasp_success)

        if cfg["task"] == 'reach':
            threshold = 0.41
        elif cfg["task"] == 'minigolf':
            threshold = 0.4
        elif cfg["task"] == 'lift':
            threshold = 0.41
        if score >= threshold: 
            success = True
            success_count += 1
        
        print(f'Sample {i} Tool head {tool_choice}: has score {score:.3f}')

        C.clear()
        if success == True: 
            if save_sample == True: 
                new_path = os.path.join("./", path_folder,"tool_" + str(tool_choice_idx), "data_" + str(tool_idx[tool_choice_idx]))
                tool_idx[tool_choice_idx] += 1
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                    
                # Downsample or upsample point clouds:
                points_2000 = pcl.upsample(2000)


                np.save(os.path.join(new_path ,"point_cloud.npy"), points_2000)
                np.save(os.path.join(new_path ,"tool_point_cloud.npy"), tool_pc)

                waypoints = [way_handle_position, 
                            way_handle_qua,
                            way_ready_position,
                            way_ready_qua,
                            way_go_position,
                            way_go_qua,
                            score]
                with open(os.path.join(new_path ,'waypoints.pkl'), 'wb') as f:
                    pickle.dump(waypoints, f)


            
        del C
    print("**************************************")
    print("Count            : ", count)
    print("Success          : ", success_count)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("**************************************")