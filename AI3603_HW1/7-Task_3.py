import sys
import os
import numpy as np
import matplotlib.pyplot as plt

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
from hybrid_a_star import run_A_Star
from hybrid_a_star.map import MapParameters, calculateMapParameters, obstaclesMap
###  END CODE HERE  ###



def self_driving_path_planner(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal.

    Arguments:
    world_map -- A 120*120 array indicating map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the start position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path.
    """

    ### START CODE HERE ###
    obstacleX, obstacleY = [], []
    start_pos = start_pos +  [np.deg2rad(0)]
    goal_pos = goal_pos + [np.deg2rad(180)]
    # load obstacles
    for i in range(world_map.shape[0]):
        for j in range(world_map.shape[1]):
            if map[i][j] == 1:
                obstacleX.append(i)
                obstacleY.append(j)
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))
    x,y,yaw = run_A_Star(start_pos, goal_pos,mapParameters)
    path = [(x[k],y[k]) for k in range(len(x))]
    # print(path)

    ###  END CODE HERE  ###
    return path




if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = self_driving_path_planner(map, start_pos, goal_pos)

    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

