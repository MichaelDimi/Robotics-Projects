#
# file   CS490_assignment2.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment2 - 
#        Motion model and landmark detection
# date   2022-08-18
#

from helper_functions import * 

#robot node class
class robot_node:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.lidar_ = None
        self.landmark_ = None
        self.pairs_ = None
        self.wall_pairs_ = None

    def add_lidar_scan(self, lidar):
        self.lidar_ = lidar

    def add_detected_landmark(self, landmark):
        self.landmark_ = landmark

    def add_landmark_pairs(self, pairs):
        self.pairs_ = pairs

#Question1
#****************************************************************************************
def location_reader(filename = None):
    locations = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            locations.append((x, y))

    return locations

def robot_motion_reader(filename = None):
    motions = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            v = float(parts[0])
            v_theta = float(parts[1])
            motions.append((v, v_theta))

    return motions

def motion_model_calculation(robot_motion):
    x, y, theta = (1, 0, 0) # Initial position
    nodes: list[robot_node] = [robot_node(x, y, theta)] # Add the initial position to the list
    dt = 0.5
    for (v, v_theta) in robot_motion:
        alpha = v_theta * dt

        if alpha == 0:
            # Straight-line motion - updates only x (move only in heading direction) 
            dx = v * dt
            dy = 0.0
            theta = theta
        else:
            R = (v * dt) / alpha
            d = R*math.sin(alpha / 2)
            dx = 2*d*math.cos(alpha / 2)
            dy = 2*d*math.sin(alpha / 2)
        # Update
        x = dx*math.cos(theta) - dy*math.sin(theta) + x
        y = dx*math.sin(theta) + dy*math.cos(theta) + y
        theta = alpha + theta
        # Store
        nodes.append(robot_node(x, y, theta))

    return nodes
#****************************************************************************************

#Question 2
#****************************************************************************************
def lidar_scan_reader(robot_node_list, filename = None):
    
    scans = []
    with open(filename, "r") as f:
        for i, ln in enumerate(f):
            parts = ln.split()
            a_min = float(parts[0])
            a_max = float(parts[1])
            d_alpha = float(parts[2])
            ranges = [min(float(x), 3.5) for x in parts[3:]]
            lidar = {
                "a_min": a_min,
                "a_max": a_max,
                "d_alpha": d_alpha,
                "ranges": ranges,
            }
            robot_node_list[i+1].add_lidar_scan(lidar)
            scans.append(lidar)
    
    return scans


def landmark_detection(robot_node_list, dthr=0.24):
    
    for i, node in enumerate(robot_node_list):
        lidar = node.lidar_
        if not lidar: # Prevent None Error
            continue

        a_min   = lidar["a_min"]
        d_alpha = lidar["d_alpha"]
        ranges  = lidar["ranges"]

        # Calculate the gradients of between rays, using the previous and next rays / 2
        n = len(ranges)
        grads = [0.0] * n
        for i in range(1, n-1):
            grads[i] = (ranges[i + 1] - ranges[i - 1]) / 2.0

        # Find landmark segments: From large NEG grad to large POS grad
        landmarks = []
        in_seg = FALSE
        start = 0 # Tracks when we enter a segment
        for i in range(1, n-1):
            g = grads[i]
            if not in_seg and g <= -dthr:
                # Entered a segment at ray index i
                in_seg = TRUE
                start = i
            elif in_seg and g >= dthr:
                # Left a segment at ray index i
                end = i
                # Take the average ray angle and distance (-offset) for each ray that hits the landmark
                mean_a = sum(a_min + k*d_alpha for k in range(start, end + 1)) / (end - start + 1)
                mean_d = sum(ranges[k] for k in range(start, end + 1)) / (end - start + 1)
                mean_d += 0.15
                # Convert from the landmark pos from robot frame to global frame
                X = node.x_ + mean_d * math.cos(node.theta_ + mean_a)
                Y = node.y_ + mean_d * math.sin(node.theta_ + mean_a)
                landmarks.append((X, Y))
                in_seg = FALSE

        # Store the landmarks
        node.add_detected_landmark(landmarks)

def plot_global_scan(robot_node_list, t, savepath=None):
    node = robot_node_list[t]
    lidar = node.lidar_
    a_min   = lidar["a_min"]
    d_alpha = lidar["d_alpha"]
    ranges  = lidar["ranges"]

    # Transform each ray endpoint to global coordinates
    scan_x, scan_y = [], []
    for i, r in enumerate(ranges):
        a = a_min + i * d_alpha
        X = node.x_ + r * math.cos(node.theta_ + a)
        Y = node.y_ + r * math.sin(node.theta_ + a)
        scan_x.append(X)
        scan_y.append(Y)

    fig, ax = plt.subplots()
    # Plot lidar endpoints
    ax.scatter(scan_x, scan_y, s=8)

    # Robot position and angle (short line in heading direction)
    L = 0.4
    hx = node.x_ + L * math.cos(node.theta_)
    hy = node.y_ + L * math.sin(node.theta_)
    ax.plot([node.x_, hx], [node.y_, hy], c="r")
    ax.scatter(node.x_, node.y_, marker="o", s=90, c="r")

    # Detected landmarks
    LX, LY = zip(*node.landmark_) # Unpack each landmark positions into two arrays
    ax.scatter(LX, LY, marker="*", s=90)

    # Show plot
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 6)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Scan & landmarks @ t={t}")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_gradient(robot_node_list, t, savepath=None, dthr=0.24):
    node = robot_node_list[t]
    lidar = node.lidar_
    a_min   = lidar["a_min"]
    d_alpha = lidar["d_alpha"]
    ranges  = lidar["ranges"]

    # Angles and gradients
    n = len(ranges)
    angles = [a_min + i * d_alpha for i in range(n)]
    grads = [0.0] * n
    for i in range(1, n - 1):
        grads[i] = (ranges[i + 1] - ranges[i - 1]) / 2.0

    # Show the plot
    fig, ax = plt.subplots()
    ax.plot(angles, grads)
    ax.axhline(dthr, c="tab:orange")
    ax.axhline(-dthr, c="tab:orange")
    ax.set_xlabel("ray angle (rad)")
    ax.set_ylabel("gradient")
    ax.set_title(f"Gradient vs Angle @ t={t}")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    
#****************************************************************************************

#Question 3
#****************************************************************************************
def pair_landmarks(robot_node_list):
    threshold = 1.0
    for node in robot_node_list:
        pairs = []
        for i, (x, y) in enumerate(node.landmark_ or []):
            # find closest groud truth landmark
            closest_j = None # closest ground truth index
            min_d2 = float("inf") # min distance (2d distance)
            for j, (gx, gy) in enumerate(gt_landmark):
                # Find Euclidean distance to ground truth
                d2 = (x - gx)**2 + (y - gy)**2
                if d2 < min_d2:
                    min_d2 = d2
                    closest_j = j
            if closest_j is not None and math.sqrt(min_d2) <= threshold:
                pairs.append((i, closest_j))
        node.add_landmark_pairs(pairs)

def plot_landmark_pairs(robot_node_list, t, savepath=None):
    node = robot_node_list[t]
    landmarks = node.landmark_ or []
    pairs = node.pairs_

    # draw matched pairs, each with a distinct color
    fig, ax = plt.subplots()
    for (i, j) in pairs:
        xd, yd = landmarks[i]
        xg, yg = gt_landmark[j]
        # draw line first to get the auto-cycle color
        line, = ax.plot([xg, xd], [yg, yd], linewidth=2)
        c = line.get_color()
        # draw both endpoints in the same color
        ax.scatter([xg, xd], [yg, yd], s=70, color=c)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 6)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"Landmark and Ground Truth pairs @ t={t}")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


#****************************************************************************************


if __name__ == '__main__':

    #you can add visualization functions but do not change the existsing code 
    #please check what you need to implement for each function in the handout

    #Question1
    #************************************************************************************
    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    # Visualize
    # draw_robot_trajectory(robot_node_list, gt_location=gt_location)
    #************************************************************************************

    #Question2
    #************************************************************************************
    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')
    landmark_detection(robot_node_list)

    # Visualize
    count = len(robot_node_list)
    # plot_global_scan(robot_node_list, 1, "LandmarkVisual1")
    # plot_global_scan(robot_node_list, int(count/2), "LandmarkVisual2")
    # plot_global_scan(robot_node_list, count-1, "LandmarkVisual3")
    # plot_gradient(robot_node_list, 1, "GradientVisual1")
    # plot_gradient(robot_node_list, int(count/2), "GradientVisual2")
    # plot_gradient(robot_node_list, count-1, "GradientVisual3")
    #************************************************************************************

    #Question3
    #************************************************************************************
    pair_landmarks(robot_node_list)

    # Visualize
    # plot_landmark_pairs(robot_node_list, 1, "PairsVisual1")
    # plot_landmark_pairs(robot_node_list, int(count/2), "PairsVisual2")
    # plot_landmark_pairs(robot_node_list, count-1, "PairsVisual3")
    #************************************************************************************



