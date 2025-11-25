#
# file   CS490_assignment3.py 
# brief  Purdue University Fall 2023 CS490 robotics Assignment3 - 
#        Integrating Motion model and Observation model for localization
# date   2023-10-1
#

from helper_functions import * 

def predict_from(pose, motion):
    x, y, theta = pose
    v, v_theta  = motion
    if v_theta != 0:
        d_theta = v_theta * 0.5
        r = v * 0.5 / abs(d_theta)
        dx = 2 * r * math.sin(abs(d_theta) / 2) * math.cos(d_theta / 2)
        dy = 2 * r * math.sin(abs(d_theta) / 2) * math.sin(d_theta / 2)
    else:
        dx = v * 0.5
        dy = 0.0
    x_pred = x + math.cos(theta) * dx - math.sin(theta) * dy
    y_pred = y + math.sin(theta) * dx + math.cos(theta) * dy
    theta_pred = theta + v_theta * 0.5
    return (x_pred, y_pred, theta_pred)

#Question 1 - feature-based localization
#****************************************************************************************
def estimate_transform(left_list, right_list, fix_scale = False):
    if left_list is None or right_list is None:
        return (1.0, 1.0, 0.0, 0.0, 0.0)
    if len(left_list) < 2 or len(right_list) < 2:
        return (1.0, 1.0, 0.0, 0.0, 0.0)
    
    # list: [(x, y)]

    # Change lists to [[x, y]]
    L = np.asarray(left_list, dtype=float)
    R = np.asarray(right_list, dtype=float)

    # Centroids
    l_bar = np.average(L, axis=0)
    r_bar = np.average(R, axis=0)
    l = L - l_bar
    r = R - r_bar

    # Find Rotations
    C = sum(lx*rx + ly*ry for (lx, ly), (rx, ry) in zip(l, r))
    S = sum(lx*ry - ly*rx for (lx, ly), (rx, ry) in zip(l, r))
    normal = sqrt(C*C + S*S)
    if normal == 0.0:
        cos, sin = 1.0, 0.0
    else:
        cos = C / normal
        sin = S / normal
    Rmat = np.array([[cos, -sin],
                  [sin,  cos]])
    
    # Scale
    l_norm = float(np.sum(l*l))
    r_norm = float(np.sum(r*r))
    if l_norm == 0: 
        lmbda = 0.0
    else :
        lmbda = (r_norm / l_norm) ** 0.5 if not fix_scale else 1

    # Translation
    t = r_bar - lmbda * (Rmat @ l_bar)
    return (lmbda, cos, sin, t[0], t[1])

def transform_estimate_and_correction(robot_node_list, robot_motion):
    
    prev_corrected = (robot_node_list[0].x_, robot_node_list[0].y_, robot_node_list[0].theta_)
    robot_node_list[0].x_, robot_node_list[0].y_, robot_node_list[0].theta_ = prev_corrected

    for i in range(1, len(robot_node_list)):
        node = robot_node_list[i]

        # Predict current pose from previous corrected pose + motion
        pred_pose = predict_from(prev_corrected, robot_motion[i-1])
        node.x_, node.y_, node.theta_ = pred_pose

        # update landmarks and pairs for the predicted position
        landmark_detection([node])
        pair_landmarks([node])

        left_list  = [node.landmark_[i] for (i, j) in node.pairs_]
        right_list = [gt_landmark[j]    for (i, j) in node.pairs_]

        T = estimate_transform(left_list, right_list, False)
        prev_corrected = correct_pose(pred_pose, T)
        node.x_, node.y_, node.theta_ = prev_corrected

#****************************************************************************************


#Question 2 - featureless localization
#****************************************************************************************
def get_subsample_rays(robot_node):
    amin = robot_node.lidar_[0]
    amax = robot_node.lidar_[1]
    delta = robot_node.lidar_[2]
    ranges = robot_node.lidar_[3:]

    x0, y0, theta = robot_node.x_, robot_node.y_, robot_node.theta_
    pts = []
    for i in range(0, len(ranges), 10):
        r = ranges[i]
        ang = amin + i*delta + theta
        x = x0 + r*math.cos(ang)
        y = y0 + r*math.sin(ang)
        pts.append((x, y))
    return pts

def get_corresponding_points_on_boundary(points):
    # World boundries
    xmin =  0.0
    xmax =  6.0
    ymin = -3.0
    ymax =  3.0

    laser_pts = []
    boundary_pts = []

    for x, y in points:
        # distances to each edge
        d_left   = abs(x - xmin)
        d_right  = abs(x - xmax)
        d_bottom = abs(y - ymin)
        d_top    = abs(y - ymax)

        # find closest edge
        dmin = min(d_left, d_right, d_bottom, d_top)
        if dmin > 0.1:
            continue  # not close enough to any boundary

        # snap to closest edge
        if dmin == d_left:
            bx, by = xmin, y
        elif dmin == d_right:
            bx, by = xmax, y
        elif dmin == d_bottom:
            bx, by = x, ymin
        else:  # dmin == d_top
            bx, by = x, ymax

        laser_pts.append((x, y))
        boundary_pts.append((bx, by))

    return laser_pts, boundary_pts

def get_icp_transform(points, iterations):    
    # Start with Identity transform
    T_total = (1.0, 1.0, 0.0, 0.0, 0.0)

    # Nudge the transform closer to local optimum repeatedly
    for _ in range(iterations):
        laser_pts, boundary_pts = get_corresponding_points_on_boundary(points)

        T_step = estimate_transform(laser_pts,
                                    boundary_pts,
                                    fix_scale=True) # Walls dont scale
        points = [apply_transform(T_step, p) for p in points]
        T_total = concatenate_transform(T_step, T_total)
    
    return T_total


def featureless_transform_estimate_and_correction(robot_node_list, robot_motion):

    prev_corrected = (robot_node_list[0].x_, robot_node_list[0].y_, robot_node_list[0].theta_)
    robot_node_list[0].x_, robot_node_list[0].y_, robot_node_list[0].theta_ = prev_corrected

    for i in range(1, len(robot_node_list)):
        node = robot_node_list[i]

        # Predict current pose from previous corrected pose + motion
        pred_pose = predict_from(prev_corrected, robot_motion[i-1])
        node.x_, node.y_, node.theta_ = pred_pose

        # Build endpoints for predicted pose
        pts = get_subsample_rays(node)

        # Run ICP
        T = get_icp_transform(pts, 100)

        # Correct the predicted pose and save it to predict the next node pose
        prev_corrected = correct_pose(pred_pose, T)
        node.x_, node.y_, node.theta_ = prev_corrected

#****************************************************************************************

if __name__ == '__main__':

    #please don't change existing code in this main function
    #check what you need to implement for each each function in the handout

    #correct implementation of all functions of assignment2 are provided
    #************************************************************************************
    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')

    robot_node_list_feature_based = copy.deepcopy(robot_node_list)

    robot_node_list_featureless = copy.deepcopy(robot_node_list)

    landmark_detection(robot_node_list_feature_based)

    pair_landmarks(robot_node_list_feature_based)
    #************************************************************************************

    #Question 1
    transform_estimate_and_correction(robot_node_list_feature_based, robot_motion)
    # Visualize
    # draw_robot_trajectory(robot_node_list_feature_based, gt_location)

    #Question 2
    featureless_transform_estimate_and_correction(robot_node_list_featureless, robot_motion)
    # Visualize
    # draw_robot_trajectory(robot_node_list_featureless, gt_location)
