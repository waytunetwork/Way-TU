import robotic as ry
import numpy as np
import time
# import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from WayTu_Rai.template_method import TemplateMethod
from sklearn.cluster import DBSCAN
import cv2
import os
        

def rotate_point(radius, angle):
    # Perform rotation using trigonometric formulas
    x_rotated = radius * np.cos(angle)
    y_rotated = radius * np.sin(angle)
    
    return x_rotated, y_rotated


def check_non_equal(old_pose, new_pose):
    change = False

    for i in range(2):
        if np.abs(old_pose[i]-new_pose[i]) > 0.03: 

            change = True
   
    return change

def IK(C,target):
    q0 = C.getJointState()
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(1., 1, 5., 0)
    komo.addControlObjective([], 0, 1e-0)
    # komo = ry.KOMO(C, 1, 1, 0, False) #one phase one time slice problem, with 'delta_t=1', order=0
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq);
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'
    # komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], qHome) #cost: close to qHome
    komo.addObjective([], ry.FS.poseDiff, ['l_gripper', target], ry.OT.eq, [1e1]) #constraint: gripper position
    
    ret = ry.NLP_Solver(komo.nlp(), verbose=0) .solve()
    
    return [komo.getPath()[0], ret]

def check_flying(C, tools, positions, max_distance = 0.1):
    print("I am in check flying")
    for tool in tools:
        correctly_positioned = False 
        for position in positions:
            tool_number = tool[4:]
            tool_name = f"tool-handle-{tool_number}"
            tool_position = np.array(C.getFrame(tool_name).getPosition())
            center = np.array(position)
            distance = np.linalg.norm(tool_position - center)
            print("distance:" , distance)
            if distance <= max_distance:
                correctly_positioned = True
                break
        if not correctly_positioned:
            return False

    return True

def tool_manipulation(C, path_method, task, tools=None, positions=None):

    pose_parameters = {}

    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(2., 1, 5., 0)
    komo.addControlObjective([], 0, 1e-0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq);
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq);

    if path_method == "Heuristic": 
        komo.addObjective([2.], ry.FS.poseDiff, ['l_gripper', 'waypoint-tool'], ry.OT.eq, [1e1]);
    elif path_method == "KOMO":
        komo.addObjective([2.], ry.FS.poseDiff, ['l_gripper', 'handle'], ry.OT.sos, [1e1]);

    ret = ry.NLP_Solver() \
    .setProblem(komo.nlp()) \
    .setOptions( stopTolerance=1e-2, verbose=0 ) \
    .solve()

    path = komo.getPath()
 

    bot = ry.BotOp(C, False)
    bot.home(C)  

    img_count = 0
    path_folder = os.path.join("./" , "simulation_frames")
    if not os.path.exists( path_folder):
        os.mkdir(path_folder)

    bot.gripperMove(ry._left, width = 0.08, speed = 0.1)

    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    time.sleep(8)
    bot.move(path, [2., 3.])
    while bot.getTimeToEnd()>0:
        bot.sync(C, .1)

    bot.gripperCloseGrasp(ry._left, 'waypoint-tool')
    while not bot.gripperDone(ry._left):
        bot.sync(C, .1)
    
    # bot.home(C)

    if task == 'reach' or task == 'lift':
        if path_method == "Heuristic":
            q_target, ret = IK(C, 'waypoint-before-ready')

        bot.moveTo(q_target, timeCost=5., overwrite=True)
        while bot.getTimeToEnd()>0:
            bot.sync(C, .1)
            if C.getFrame('l_gripper').getPosition()[2] > 5.0:
                dance = True

    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(2., 1, 5., 0)
    komo.addControlObjective([], 0, 1e-0)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq);
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq);

    if path_method == "Heuristic":

        komo.addObjective([2.], ry.FS.poseDiff, ['l_gripper', 'waypoint-ready'], ry.OT.eq, [1e2]);
        # komo.addObjective([4.], ry.FS.poseDiff, ['l_gripper', 'way_go'], ry.OT.eq, [1e1]);
        # q_target, ret = IK(C, 'way_ready')
    elif path_method == "KOMO":
        # komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', 'way_ready'], ry.OT.eq, [1e1]);
        komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', 'ball-middle'], ry.OT.eq, [1e1]);
        komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', 'ball-left'], ry.OT.eq, [1e1]);
        komo.addObjective([1.], ry.FS.poseDiff, ['l_gripper', 'ball-right'], ry.OT.eq, [1e1]);
        komo.addObjective([2.], ry.FS.poseDiff, ['l_gripper', 'way_go'], ry.OT.eq, [1e1]);
    
    ret = ry.NLP_Solver() \
    .setProblem(komo.nlp()) \
    .setOptions( stopTolerance=1e-2, verbose=0 ) \
    .solve()

    path = komo.getPath()
    
    dance = False

    bot.move(path, [2.,3.])

    while bot.getTimeToEnd()>0:
        bot.sync(C, .1)
        if C.getFrame('l_gripper').getPosition()[2] > 5.0:
            dance = True

    if path_method == "Heuristic":
        q_target, ret = IK(C, 'waypoint-middle-1')

    bot.moveTo(q_target, timeCost=5., overwrite=True)
    while bot.getTimeToEnd()>0:
        bot.sync(C, .1)
        if C.getFrame('l_gripper').getPosition()[2] > 5.0:
            dance = True

    # Open for minigolf
    if task =='minigolf': 
        if path_method == "Heuristic":
            q_target, ret = IK(C, 'waypoint-middle-2')
        
        bot.moveTo(q_target, timeCost=5., overwrite=True)
        while bot.getTimeToEnd()>0:
            bot.sync(C, .1)
            if C.getFrame('l_gripper').getPosition()[2] > 5.0:
                dance = True


    if path_method == "Heuristic":
        q_target, ret = IK(C, 'waypoint-go')

    bot.moveTo(q_target, timeCost=5., overwrite=True)
    while bot.getTimeToEnd()>0:
        bot.sync(C, .1)
        if C.getFrame('l_gripper').getPosition()[2] > 5.0:
            dance = True


    if task =="push":
        pose_parameters["new_middle"] = np.array(C.getFrame('ball-middle').getPosition())
        pose_parameters["new_left"] = np.array(C.getFrame('ball-left').getPosition())
        pose_parameters["new_right"] = np.array(C.getFrame('ball-right').getPosition())

    del bot

    return pose_parameters, dance

def check_success (pose_parameters, dance):
    if dance == True: 
        return 0, 0
    else:
        middle_check = 0
        triple_check = 0 
        if check_non_equal(pose_parameters["old_middle"], pose_parameters["new_middle"]):
            middle_check = 1
            if check_non_equal(pose_parameters["old_left"], pose_parameters["new_left"]) and check_non_equal(pose_parameters["old_right"], pose_parameters["new_right"]):
                triple_check = 1
        return middle_check, triple_check 