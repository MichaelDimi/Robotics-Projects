#
# file   helper_functions.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment3 - 
#        Integrating Motion model and Observation model for localization help functions
# date   2022-08-18
#

import os
import sys
import math
import copy
from math import sqrt, atan2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

gt_landmark = [[1, 2], [4, 1], [2, 0], [5, -1], [1, -2], [3, -2]]

#helper class
class robot_node:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.lidar_ = None

    def add_lidar_scan(self, lidar):
        self.lidar_ = lidar

    def add_detected_landmark(self, landmark):
        self.landmark_ = landmark

    def add_landmark_pairs(self, pairs):
        self.pairs_ = pairs

#Solution code for assignment2, now they are provided as the starting point of assignment3
#****************************************************************************************
#read ground truth robot motion
def location_reader(filename = None):
    location = []
    if filename:
        raw_data = open(filename, 'r')
        for line in raw_data.readlines():
            processed_data = [float(x) for x in line[:-1].split()]
            location.append(tuple(processed_data))
    return location

def robot_motion_reader(filename = None):
    robot_motion = []
    if filename:
         raw_data = open(filename, 'r')
         for line in raw_data.readlines():
             processed_data = [float(x) for x in line[:-1].split()]
             robot_motion.append(tuple(processed_data))
    return robot_motion

def motion_model_calculation(robot_motion):
    motion_model = []
    motion_model.append(robot_node(1, 0, 0))
    if robot_motion:
        for v, v_theta in robot_motion:
            if v_theta != 0:
                delta_theta = v_theta * 0.5
                r = v * 0.5 / abs(delta_theta)
                delta_x, delta_y = None, None
                #if v_theta >= 0:
                delta_x = 2 * r * math.sin(abs(delta_theta) / 2) * math.cos(delta_theta / 2)
                delta_y = 2 * r * math.sin(abs(delta_theta) / 2) * math.sin(delta_theta / 2)
                #else:
                #    delta_x = - 2 * r * math.sin(delta_theta / 2) * math.cos(delta_theta )
                #    delta_y = - 2 * r * math.sin(delta_theta / 2) * math.sin(delta_theta )
            else:
                delta_x = v * 0.5
                delta_y = 0
            p_node = motion_model[-1]
            ori_x, ori_y, ori_theta = p_node.x_, p_node.y_, p_node.theta_

            final_theta = ori_theta + v_theta * 0.5
            final_x = ori_x + math.cos(ori_theta) * delta_x - math.sin(ori_theta) * delta_y
            final_y = ori_y + math.sin(ori_theta) * delta_x + math.cos(ori_theta) * delta_y
            motion_model.append(robot_node(final_x, final_y, final_theta))
    if motion_model:
        return motion_model[:-1]
    else:
        return motion_model

def lidar_scan_reader(robot_node_list, filename = None):
    if filename:
        counter = 0
        raw_data = open(filename, 'r')
        for line in raw_data.readlines():
            processed_data = [float(x) for x in line[:-1].split()]
            processed_data = [x if x != 10.0 else 3.5 for x in processed_data]
            robot_node_list[counter].lidar_ = processed_data
            counter += 1

def landmark_detection(robot_node_list):

    for t in range(len(robot_node_list)):
        robot_node = robot_node_list[t]
        single_lidar_scan = robot_node.lidar_

        min_angle, max_angle, delta_angle = single_lidar_scan[:3]

        scan_rays = single_lidar_scan[3:]
        scan_rays_copy = scan_rays[:]
        scan_rays = [scan_rays[-1]] + scan_rays + [scan_rays[0]]
        
        gradient_data = []

        for i in range(1, len(scan_rays)-1):
            l = scan_rays[i-1]
            r = scan_rays[i+1]
            derivative = (r - l) / 2.0
            gradient_data.append(derivative)

        gradient_data = gradient_data + gradient_data

        detected_landmark = []

        ray_index_sum, ray_distance_sum, ray_sum = 0, 0, 0
        inside_landmark = False

        for i in range(len(gradient_data)):
            measure = gradient_data[i]
            if measure < -0.24:
                if i >= 360: break
                ray_index_sum, ray_distance_sum, ray_sum = 0, 0, 0
                inside_landmark = True
            elif measure > 0.24:
                if inside_landmark and ray_sum:
                    detected_landmark.append((ray_index_sum / ray_sum, ray_distance_sum / ray_sum))
                inside_landmark = False
                if i >= 360:
                    break
            else:
                if inside_landmark:
                    ray_index_sum += i
                    if i < 360:
                        ray_distance_sum += scan_rays_copy[i]
                    else:
                        ray_distance_sum += scan_rays_copy[i - 360]
                    ray_sum += 1

        landmarks = []
        for ray_index, ray_length in detected_landmark:
            #offset
            ray_length += 0.15
            ray_angle = ray_index * delta_angle
            ray_angle += robot_node.theta_
            landmarks.append((robot_node.x_ + math.cos(ray_angle) * ray_length, robot_node.y_ + math.sin(ray_angle) * ray_length))

        robot_node.add_detected_landmark(landmarks)

def pair_landmarks(robot_node_list):
    
    for robot_node in robot_node_list:
        landmark = robot_node.landmark_

        landmark_pairs = []

        max_radius = 1.0

        for i in range(len(landmark)):
            cx, cy = landmark[i]
            distance = []
            for rx, ry in gt_landmark:
                temp_distance = ((cx - rx)**2 + (cy - ry)**2)
                if temp_distance < max_radius**2:
                    distance.append(temp_distance)
                else:
                    distance.append(sys.maxsize)
            min_dis = min(distance)
            if min_dis != sys.maxsize:
                landmark_pairs.append((i, distance.index(min_dis)))

        robot_node.add_landmark_pairs(landmark_pairs)
#****************************************************************************************

#new helper functions that you can use for assignment 3
#****************************************************************************************
#concat two transforms
def concatenate_transform(a, b):
    laa, ca, sa, txa, tya = a
    lab, cb, sb, txb, tyb = b

    la = laa * lab

    c = ca*cb - sa*sb
    s = sa*cb + ca*sb

    tx = txa + laa * ca * txb - laa * sa * tyb
    ty = tya + laa * sa * txb + laa * ca * tyb

    return (la, c, s, tx, ty)

# the center of mass computation
def compute_center(point_list):
    if not point_list:
        return (0.0, 0.0)
    sx = sum([p[0] for p in point_list])
    sy = sum([p[1] for p in point_list])
    return (sx / len(point_list), sy / len(point_list))

def apply_transform(trafo, p):
    la, c, s, tx, ty = trafo
    lac = la * c
    las = la * s
    x = lac * p[0] - las * p[1] + tx
    y = las * p[0] + lac * p[1] + ty
    return (x, y)

#correct robot status
def correct_pose(pose, trafo):

    la, c, s, tx, ty = trafo

    x, y, theta = pose

    new_x, new_y = apply_transform(trafo, (x, y))

    alpha = atan2(s, c)

    theta += alpha

    return (new_x, new_y, theta)

#visualization function
def draw_robot_trajectory(robot_node_list = None, gt_location = None):
    if robot_node_list:
        window = Tk()

        window.geometry('700x700')

        canvas = Canvas(window, width = 600, height = 600)

        canvas.pack()

        if gt_location:
            for i in range(len(gt_location)-1):
                x0, y0 = gt_location[i]
                x1, y1 = gt_location[i+1]

                x0 = int(x0 * 100)
                y0 = 600 - int(y0 * 100) - 300
                x1 = int(x1 * 100)
                y1 = 600 - int(y1 * 100) - 300

                canvas.create_line(x0, y0, x1, y1, fill = 'blue', width = 3)

        for i in range(len(robot_node_list)-1):
            x0, y0 = robot_node_list[i].x_, robot_node_list[i].y_
            x1, y1 = robot_node_list[i+1].x_, robot_node_list[i+1].y_

            x0 = int(x0 * 100)
            y0 = 600 - int(y0 * 100) - 300
            x1 = int(x1 * 100)
            y1 = 600 - int(y1 * 100) - 300

            canvas.create_line(x0, y0, x1, y1, fill = 'red', width = 3)

        

        window.mainloop()
#****************************************************************************************

