"""
Path Planning Sample Code with RRT*

author: Ahmed Qureshi, code adapted from AtsushiSakai(@Atsushi_twi)

"""


import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2))

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, alg, geom, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start)
        self.end = Node(goal)
        self.obstacleList = obstacleList
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.alg = alg
        self.geom = geom
        self.dof = dof

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.goalfound = False
        self.solutionSet = set()

    def planning(self, animation=False):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):
            rnd = self.generatesample()
            nind = self.GetNearestListIndex(self.nodeList, rnd)


            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])


            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar':
                    self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True

                if animation:
                    self.draw_graph(rnd.state)

        return self.get_path_to_goal()

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        # your code here

        # Current best is the nearest node alreaedy chosen in planning()
        best_parent = newNode.parent
        best_cost = newNode.cost

        for i in nearinds:
            parent_node = self.nodeList[i]

            # Check if we can connect parent_node -> newNode withou collision
            success, edge_cost = self.steerTo(newNode, parent_node)
            if not success:
                continue

            # Cost to reach newNode through this candidate parent
            candidate_cost = parent_node.cost + edge_cost

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_parent = i

        # If nothing beat the existing parent, signal no change to tree
        if best_parent == newNode.parent:
            return None

        return best_parent
        


    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        This function is used in planning() to filter out invalid random samples. You may also find it useful
        for implementing the functions in question 1.

        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)
        for j in range(0,self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)


        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0,self.dof):
                dists[j] =dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1

            stateCurr = np.zeros(self.dof,dtype=np.float32)
            for j in range(0,self.dof):
                stateCurr[j] = newNode.state[j]

            stateCurr = Node(stateCurr)

            for i in range(0,numSegments):

                if not self.__CollisionCheck(stateCurr):
                    return (False, None)

                for j in range(0,self.dof):
                    stateCurr.state[j] += dists[j]

            if not self.__CollisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def generatesample(self):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """
        if random.randint(0, 100) > self.goalSampleRate:
            if self.geom == 'rectangle':
                # (x, y) in workspace, theta in [-pi, pi]
                x = random.uniform(self.minrand, self.maxrand)
                y = random.uniform(self.minrand, self.maxrand)
                theta = random.uniform(-math.pi, math.pi)
                rnd = Node([x, y, theta])
            else:
                sample=[]
                for j in range(0,self.dof):
                    sample.append(random.uniform(self.minrand, self.maxrand))
                rnd=Node(sample)
        else:
            rnd = self.end
        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = dist(node.state, self.end.state)
        if d < 5.0:
            return True
        return False

    @staticmethod
    def get_path_len(path):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += dist(path[i], path[i-1])

        return pathLen


    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.end.state]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append(node.state)
            goalind = node.parent
        path.append(self.start.state)
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        # Use this value of gamma
        GAMMA = 50

        # your code here
        # Number of existing nodes in the tree
        num_nodes = len(self.nodeList)

        # If the tree is empty, nothing is near
        if num_nodes == 0:
            return []

        # Compute the RRT* neighborhood radius:
        radius = GAMMA * (math.log(num_nodes) / num_nodes) ** (1.0 / float(self.dof))

        near = []
        for i, node in enumerate(self.nodeList):
            if dist(newNode.state, node.state) <= radius:
                near.append(i)

        return near

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
        Recall that a node should be rewired if doing so would reduce its cost.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        # your code here

        # Compute the ancestor chain of newNode (start -> ... -> parent -> newNode).
        ancestors = set()
        current = newNode.parent
        while current is not None:
            ancestors.add(current)
            current = self.nodeList[current].parent

        for idx in nearinds:
            # Skip the new node itself
            if idx == newNodeIndex:
                continue

            # Do not rewire ancestors of newNode, to avoid cycles.
            if idx in ancestors:
                continue

            near_node = self.nodeList[idx]

            # Check if the edge newNode -> near_node is collision-free
            success, edge_cost = self.steerTo(near_node, newNode)
            if not success:
                continue

            # Cost to reach near_node via newNode
            new_cost = newNode.cost + edge_cost

            # Only rewire if this is strictly better
            if new_cost < near_node.cost:
                old_cost = near_node.cost
                old_parent_index = near_node.parent

                # Update parent/children relationships
                if old_parent_index is not None:
                    self.nodeList[old_parent_index].children.discard(idx)

                near_node.parent = newNodeIndex
                self.nodeList[newNodeIndex].children.add(idx)

                # Update near_node's cost and propagate the improvement to its subtree
                delta = new_cost - old_cost
                near_node.cost = new_cost
                self._propagate_cost_change(idx, delta) # Recursively propagate the cost updates to the subtree

    def GetNearestListIndex(self, nodeList, rnd):
        """
        Searches nodeList for the closest vertex to rnd

        nodeList: list of all nodes currently in the tree
        rnd: node to be added (not currently in the tree)

        Returns: index of nearest node
        """
        dlist = []
        for node in nodeList:
            dlist.append(dist(rnd.state, node.state))

        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node):
        """
        Checks whether a given configuration is valid. (collides with obstacles)

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        s = np.zeros(2, dtype=np.float32)
        s[0] = node.state[0]
        s[1] = node.state[1]

        if self.geom == 'circle':
            # Robot is a circle of radius 1
            radius = 1.0
            r2 = radius * radius

            for (ox, oy, sizex, sizey) in self.obstacleList:
                # Obstacles are axis-aligned rectangles: [ox, ox+sizex] x [oy, oy+sizey]
                # Find the closest point on the rectangle to the circle center
                closest_x = min(max(s[0], ox), ox + sizex)
                closest_y = min(max(s[1], oy), oy + sizey)

                dx = s[0] - closest_x
                dy = s[1] - closest_y
                dist2 = dx * dx + dy * dy

                # If the distance from center to rectangle is <= radius,
                # the circle intersects the obstacle -> Collision
                if dist2 <= r2:
                    return False

            return True 
        
        if self.geom == 'rectangle':
            theta = node.state[2]

            # Robot is a box of width 3 (long side) and height 1.5 (short side)
            robot_poly = self._rectangle_corners(s[0], s[1], theta, width=3.0, height=1.5)

            for (ox, oy, sizex, sizey) in self.obstacleList:
                # Axis-aligned obstacle box
                obs_poly = [
                    (ox,         oy),
                    (ox + sizex, oy),
                    (ox + sizex, oy + sizey),
                    (ox,         oy + sizey),
                ]
                if self._polygons_intersect(robot_poly, obs_poly):
                    return False  # collision
            return True 

        else:
            for (ox, oy, sizex,sizey) in self.obstacleList:
                obs=[ox+sizex/2.0,oy+sizey/2.0]
                obs_size=[sizex,sizey]
                cf = False
                for j in range(self.dof):
                    if abs(obs[j] - s[j])>obs_size[j]/2.0:
                        cf=True
                        break
                if cf == False:
                    return False
            return True  # safe'''

    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        if self.goalfound:
            goalind = None
            mincost = float('inf')
            for idx in self.solutionSet:
                cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    def draw_graph(self, rnd=None):
        """
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for (ox, oy, sizex, sizey) in self.obstacleList:
            rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
            plt.gca().add_patch(rect)

        for node in self.nodeList:
            if node.parent is not None:
                if node.state is not None:
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
                        node.state[1], self.nodeList[node.parent].state[1]], "-g")

        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0] for p in path]
            y = [p[1] for p in path]

            if self.geom == 'circle':
                # Draw the path of the circle's center
                plt.plot(x, y, '-r')
                radius = 1.0
                # Draw the actual circular body along the path
                for px, py in path:
                    circ = mpatches.Circle((px, py), radius,
                                           fill=False, color="red", linewidth=0.5)
                    plt.gca().add_patch(circ)

            elif self.geom == 'rectangle':
                plt.plot(x, y, '-r')
                width = 3.0
                height = 1.5
                for state in path:
                    px, py, theta = state
                    corners = self._rectangle_corners(px, py, theta, width, height)
                    poly = mpatches.Polygon(corners, fill=False,
                                            edgecolor="red", linewidth=0.5)
                    plt.gca().add_patch(poly)

            else:
                plt.plot(x, y, '-r')

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")

        # Mark centers of start and goal
        if self.geom == 'circle':
            radius = 1.0
            start_circ = mpatches.Circle((self.start.state[0], self.start.state[1]),
                                         radius, fill=False, color="red")
            goal_circ = mpatches.Circle((self.end.state[0], self.end.state[1]),
                                        radius, fill=False, color="red")
            plt.gca().add_patch(start_circ)
            plt.gca().add_patch(goal_circ)
            plt.plot(self.start.state[0], self.start.state[1], "xr")
            plt.plot(self.end.state[0], self.end.state[1], "xr")
        elif self.geom == 'rectangle':
            width = 3.0
            height = 1.5
            sc = self._rectangle_corners(self.start.state[0],
                                         self.start.state[1],
                                         self.start.state[2],
                                         width, height)
            sg = self._rectangle_corners(self.end.state[0],
                                         self.end.state[1],
                                         self.end.state[2],
                                         width, height)
            start_poly = mpatches.Polygon(sc, fill=False, edgecolor="red")
            goal_poly = mpatches.Polygon(sg, fill=False, edgecolor="red")
            plt.gca().add_patch(start_poly)
            plt.gca().add_patch(goal_poly)
            plt.plot(self.start.state[0], self.start.state[1], "xr")
            plt.plot(self.end.state[0], self.end.state[1], "xr")
        else: 
            plt.plot(self.start.state[0], self.start.state[1], "xr")
            plt.plot(self.end.state[0], self.end.state[1], "xr")

        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.pause(0.01)

    # Helpers:
    def _propagate_cost_change(self, node_index, delta_cost): # Used in re-wiring
        node = self.nodeList[node_index]
        for child_index in node.children:
            child = self.nodeList[child_index]
            child.cost += delta_cost
            self._propagate_cost_change(child_index, delta_cost)

    # The following two are used for SAT in collision checking for the rectangular case
    def _rectangle_corners(self, cx, cy, theta, width, height):
        """
        Returns the 4 world-space corners of a rectangle of size (width x height)
        centered at (cx, cy) and rotated by theta (radians).
        """
        hw = width / 2.0
        hh = height / 2.0

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Local coordinates: corners of axis-aligned rectangle
        local = [
            ( hw,  hh),
            (-hw,  hh),
            (-hw, -hh),
            ( hw, -hh),
        ]

        corners = []
        for lx, ly in local:
            wx = cx + cos_t * lx - sin_t * ly
            wy = cy + sin_t * lx + cos_t * ly
            corners.append((wx, wy))
        return corners

    def _polygons_intersect(self, poly1, poly2):
        """
        Separating Axis Theorem for convex polygons in 2D.
        Returns True if poly1 and poly2 intersect.
        """
        def get_axes(poly):
            axes = []
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                edge = (x2 - x1, y2 - y1)
                # Perpendicular (normal) to the edge
                normal = (-edge[1], edge[0])
                length = math.hypot(normal[0], normal[1])
                if length == 0:
                    continue
                axes.append((normal[0] / length, normal[1] / length))
            return axes

        def project(poly, axis):
            ax, ay = axis
            dots = [x * ax + y * ay for (x, y) in poly]
            return min(dots), max(dots)

        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                # Found a separating axis -> no collision
                return False
        # Overlap on all axes -> polygons intersect
        return True

class Node():
    """
    RRT Node
    """

    def __init__(self,state):
        self.state =state
        self.cost = 0.0
        self.parent = None
        self.children = set()



def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'], \
        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrt', choices=['rrt', 'rrtstar'], \
        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')

    args = parser.parse_args()

    show_animation = not args.blind and not args.fast

    print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
    starttime = time.time()


    obstacleList = [
    (-15,0, 15.0, 5.0),
    (15,-10, 5.0, 10.0),
    (-10,8, 5.0, 15.0),
    (3,15, 10.0, 5.0),
    (-10,-10, 10.0, 5.0),
    (5,-5, 5.0, 5.0),
    ]

    if args.geom == 'rectangle':
        start = [-10.0, -17.0, 0.0]
        goal  = [10.0, 10.0, 0.0]
        dof   = 3
    else:
        start = [-10.0, -17.0]
        goal  = [10.0, 10.0]
        dof   = 2

    rrt = RRT(start=start, goal=goal, randArea=[-20, 20], obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter)
    path = rrt.planning(animation=show_animation)

    endtime = time.time()

    if path is None:
        print("FAILED to find a path in %.2fsec"%(endtime - starttime))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec"%(RRT.get_path_len(path), endtime - starttime))
    # Draw final path
    if not args.blind:
        rrt.draw_graph()
        plt.show()


if __name__ == '__main__':
    main()
