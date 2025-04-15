from numpy import cos, sin, deg2rad, round, sqrt
import time
import heapq
import cv2
import numpy as np

# Define canvas dimensions (height and width)
canvas_height, canvas_width = 300, 540
wheel_radius = 3.3
wheel_distance = int(290 / 100) * 10
robot_radius = int(220 / 100) * 10

# Define the Node class to store information about each node in the search space
class Node:
    """Class to store the node information."""
    def __init__(self, coords, cost, parent=None, heuristic=0):
        # Coordinates of the node (x, y, theta)
        self.coords = coords  # Node's position (x, y, theta)
        self.x = coords[0]  # x-coordinate
        self.y = coords[1]  # y-coordinate
        self.orientation = coords[2]  # Orientation (theta)
        self.cost = cost  # Cost to reach this node from the start
        self.parent = parent  # Parent node for backtracking
        self.heuristic = heuristic  # Heuristic cost to the goal

    def __lt__(self, other):
        """Comparison operator for priority queue."""
        return self.cost + self.heuristic < other.cost + other.heuristic

# Define obstacle-checking functions for different shapes (letters and numbers)
def is_in_obstacle_rect1(x, y, x0=100, y0=200, width=10, height=200, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    if x0 <= x <= x0 + thickness and canvas_height - y0 <= y <= canvas_height:
        return True
    return False

def is_in_obstacle_rect2(x, y, x0=210, y0=300, width=10, height=200, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    if x0 <= x <= x0 + thickness and canvas_height - y0 <= y <= canvas_height:
        return True
    return False

def is_in_obstacle_rect3(x, y, x0=320, y0=300, width=10, height=100, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    if x0 <= x <= x0 + thickness and canvas_height - y0 <= y <= canvas_height:
        return True
    return False

def is_in_obstacle_rect4(x, y, x0=320, y0=100, width=10, height=100, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    if x0 <= x <= x0 + thickness and canvas_height - y0 <= y <= canvas_height:
        return True
    return False

def is_in_obstacle_rect5(x, y, x0=430, y0=200, width=10, height=200, thickness=3):
    """Checks if a point (x, y) is inside the number '1'."""
    if x0 <= x <= x0 + thickness and canvas_height - y0 <= y <= canvas_height:
        return True
    return False

def generate_map(clearance):
    """Creates the map with obstacles and clearance zones."""
    grid = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    clearance_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    offset = int(clearance + robot_radius)

    for i in range(canvas_height):
        for j in range(canvas_width):
            if (
                is_in_obstacle_rect1(j, i)
                or is_in_obstacle_rect2(j, i)
                or is_in_obstacle_rect3(j, i)
                or is_in_obstacle_rect4(j, i)
                or is_in_obstacle_rect5(j, i)
            ):
                grid[i][j] = 1  # Mark as obstacle

    kernel = np.ones((int(offset * 2 + 1), int(offset * 2 + 1)), np.uint8)
    clearance_mask = cv2.dilate(grid, kernel, iterations=1)
    clearance_mask[np.where(grid == 1)] = 2
    clearance_mask[0:offset, :] =                                                               1
    clearance_mask[canvas_height - offset : canvas_height, :] = 1
    clearance_mask[:, 0:offset] = 1
    clearance_mask[:, canvas_width - offset : canvas_width] = 1
    grid = clearance_mask
    return grid, clearance_mask

def is_valid_point(x, y, grid):
    """Checks whether the given coordinates are valid (not on an obstacle)."""
    if grid[y][x] == 1:
        return False
    return True

def round_value(number):
    """Rounds the given number to nearest 0.5."""
    return np.around(number * 2.0) / 2.0

def differential_drive_action(x, y, theta, UL, UR, dt=0.05, steps=20):
    """Simulates motion of a differential drive robot using wheel RPMs."""
    t = 0
    theta = deg2rad(theta)
    Xn, Yn, Thetan = x, y, theta
    path = []

    for _ in range(steps):
        t += dt
        ul_rad = (UL * 2 * np.pi) / 60
        ur_rad = (UR * 2 * np.pi) / 60
        dx = 0.5 * wheel_radius * (ul_rad + ur_rad) * np.cos(Thetan) * dt
        dy = 0.5 * wheel_radius * (ul_rad + ur_rad) * np.sin(Thetan) * dt
        dtheta = (wheel_radius / wheel_distance) * (ur_rad - ul_rad) * dt
        Xn += dx
        Yn += dy
        Thetan += dtheta
        path.append((Xn, Yn))

    return [Xn, Yn, np.rad2deg(Thetan)], path

def generate_successors(node, RPM1_val, RPM2_val, canvas):
    """Generates all valid successors based on differential drive actions."""
    x, y, theta = node.coords
    cost = node.cost
    successors = []
    rpm_pairs = [
        (0, RPM1_val),
        (RPM1_val, 0),
        (RPM1_val, RPM1_val),
        (0, RPM2_val),
        (RPM2_val, 0),
        (RPM2_val, RPM2_val),
        (RPM1_val, RPM2_val),
        (RPM2_val, RPM1_val),
    ]

    for ul, ur in rpm_pairs:
        new_coords, path = differential_drive_action(x, y, theta, ul, ur)
        xn, yn, thetan = new_coords
        if 0 <= xn < canvas_width and 0 <= yn < canvas_height:
            x_idx = min(int(round(xn)), canvas_width - 1)
            y_idx = min(int(round(yn)), canvas_height - 1)
            if canvas[y_idx][x_idx] == 0:
                rounded_coords = [round_value(xn), round_value(yn), thetan % 360]
                new_cost = cost + np.linalg.norm([xn - x, yn - y])
                successors.append([rounded_coords, new_cost])

    return successors

def check_visited(x, y, theta, visited_mat, visited_threshold):
    """Checks duplicate nodes by discretizing x, y, and theta."""
    xn = int(round_value(x) / visited_threshold)
    yn = int(round_value(y) / visited_threshold)
    xn = min(max(xn, 0), visited_mat.shape[1] - 1)
    yn = min(max(yn, 0), visited_mat.shape[0] - 1)
    theta_mod = theta % 360
    num_bins = visited_mat.shape[2]
    thetan = int(theta_mod / (360 / num_bins))
    thetan = min(max(thetan, 0), num_bins - 1)

    if visited_mat[yn][xn][thetan] == 0:
        visited_mat[yn][xn][thetan] = 1
        return True
    return False

def astar_search(start_node, goal_node, RPM1_val, RPM2_val, canvas, visited_mat, visited_threshold, goal_threshold):
    """Performs A* search to find the shortest path."""
    node_graph = {}
    open_list = {}
    closed_list = {}
    queue = []

    open_list[str([start_node.x, start_node.y])] = start_node
    heapq.heappush(queue, [start_node.cost, start_node])

    while len(queue) != 0:
        fetched_ele = heapq.heappop(queue)
        current_node = fetched_ele[1]

        node_graph[str([current_node.x, current_node.y])] = current_node

        if sqrt((current_node.x - goal_node.x) ** 2 + (current_node.y - goal_node.y) ** 2) < goal_threshold:
            goal_node.parent = current_node.parent
            goal_node.cost = current_node.cost
            print("Found the goal node")
            break

        if str([current_node.x, current_node.y]) in closed_list:
            continue
        else:
            closed_list[str([current_node.x, current_node.y])] = current_node

        del open_list[str([current_node.x, current_node.y])]

        child_list = generate_successors(current_node, RPM1_val, RPM2_val, canvas)

        for child in child_list:
            child_x, child_y, child_theta = child[0]
            child_cost = child[1]

            if str([child_x, child_y]) in closed_list:
                continue

            child_heuristic = sqrt((goal_node.x - child_x) ** 2 + (goal_node.y - child_y) ** 2)
            child_node = Node([child_x, child_y, child_theta], child_cost, current_node, child_heuristic)

            if check_visited(child_x, child_y, child_theta, visited_mat, visited_threshold):
                if str([child_x, child_y]) in open_list:
                    if child_node.cost < open_list[str([child_x, child_y])].cost:
                        open_list[str([child_x, child_y])].cost = child_cost
                        open_list[str([child_x, child_y])].parent = current_node
                else:
                    open_list[str([child_x, child_y])] = child_node
                    heapq.heappush(queue, [(child_cost + child_heuristic), child_node])
    return node_graph

def backtrack_path(node_graph, goal_node):
    """Backtracks to find the path from start to goal."""
    path = []
    path.append([int(goal_node.x), int(goal_node.y), int(goal_node.orientation)])
    parent = list(node_graph.items())[-1][1].parent

    while parent:
        path.append([int(parent.x), int(parent.y), int(parent.orientation)])
        parent = parent.parent
    return path

def plan_path(start=(50, 50, 0),
    end=(500, 150, 0),
    robot_radius=22,
    clearance=2,
    delta_time=0.2,
    goal_threshold=2,
    wheel_radius=3.5,
    wheel_distance=23.0,
    rpm1=5,
    rpm2=10):
    """
    Plans a path from the start point to the goal point using the A* algorithm.
    
    Parameters:
        start (tuple): Start point as (x, y, theta).
        end (tuple): Goal point as (x, y).
        clearance (int): Clearance value in mm.
        RPM1 (int): Left wheel RPM.
        RPM2 (int): Right wheel RPM.
    
    Returns:
        list: Generated path as a list of [x, y, theta].
    """
    
    global canvas, canvas_height, canvas_width
    visited_threshold = 0.5
    visited_mat = np.zeros((500, 1200, 12), dtype=np.uint8)
    goal_threshold = 1.5

    import rclpy
    from rclpy.logging import get_logger
    logger = get_logger("my_logger")
    start_x, start_y, start_theta = start
    goal_x, goal_y,_ = end
    logger.info(f"Start xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: {start_x}")
    logger.info(f"Start yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: {start_y}")
    logger.info(f"Goal  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: {goal_x}")
    logger.info(f"Goal  yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy: {goal_y}")
    if (
        start_x > canvas_width
        or start_y > canvas_height
        or goal_x > canvas_width
        or goal_y > canvas_height
    ):
        raise ValueError("Start or goal point is out of bounds.")

    canvas, clearance_mask = generate_map(clearance)
    start_y = canvas_height - start_y - 1
    goal_y = canvas_height - goal_y - 1

    start_node = Node([start_x, start_y, start_theta], 0)
    goal_node = Node([goal_x, goal_y, 0], 0)

    print("Finding the goal node...")
    start_time = time.time()
    node_graph = astar_search(start_node, goal_node, rpm1, rpm2, canvas, visited_mat, visited_threshold, goal_threshold)
    final_path = backtrack_path(node_graph, goal_node)
    end_time = time.time()

    print(f"The output was processed in {end_time - start_time:.2f} seconds.")
    print("------------------------------------------")

    path_gen = final_path[::-1]
    canvas_height = 300.0 
    prev_x, prev_y, prev_theta = 50.0, 50.0, 0.0
    path = []
    for x_cm, y_cm, theta in path_gen:
        # Transform x and y from cm to m
        goal_x = x_cm
        goal_y = y_cm 
        # goal_y = canvas_height - goal_y
        goal_y =  (canvas_height)-goal_y 
        goal_theta = theta * (3.14 / 180)

        # Calculate deltas
        delta_x = goal_x - prev_x
        delta_y = -( goal_y - prev_y )
        delta_theta = goal_theta - prev_theta

        path.append([delta_x, delta_y, delta_theta])

        # Update previous values
        prev_x, prev_y, prev_theta = goal_x, goal_y, goal_theta

    return path