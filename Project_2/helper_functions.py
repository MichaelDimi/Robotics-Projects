#
# file   helper_functions.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment2 - 
#        Motion model and landmark detection helper functions
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

#visualization function
def draw_robot_trajectory(robot_node_list, gt_location = None):
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

