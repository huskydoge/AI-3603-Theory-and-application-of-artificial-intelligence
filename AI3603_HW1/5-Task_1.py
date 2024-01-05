import sys
import os
import numpy as np
import matplotlib.pyplot as plt

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

class A_Star_Map:
    def __init__(self, world_map, start_pos=(10, 10), goal_pos=(100, 100)):
        self.world_map = world_map
        self.map_x, self.map_y = np.shape(world_map)  # 120, 120

        if type(start_pos) is not tuple or type(goal_pos) is not tuple:
            print('start_pos and goal_pos should be tuple')
            self.start_pos = (start_pos[0], start_pos[1])
            self.goal_pos = (goal_pos[0], goal_pos[1])
        else:
            self.start_pos = start_pos
            self.goal_pos = goal_pos

        self.open_list = [self.start_pos]  # start_pos (x,y)
        self.closed_list = []

        self.g = dict()  # dict((x,y): value), cost of the path from start to the current node
        self.g[self.start_pos] = 0  # g(start) = 0

        self.h = dict()  # dict((x,y): value), heuristic function
        self.init_h()

        self.parent_nodes = dict()  # dict((x,y): (x1,y1))
        self.parent_nodes[self.start_pos] = self.start_pos
        self.path = []

    def get_4_neighbours(self, pos):
        neighbours = []
        for i in [-1, 1]:
            neighbour = (pos[0] + i, pos[1])
            if neighbour[0] < 0 or neighbour[0] >= 120 or neighbour[1] < 0 or neighbour[1] >= 120:
                continue
            if self.world_map[neighbour[0]][neighbour[1]] == 1:
                continue
            neighbours.append(neighbour)
        for j in [-1, 1]:
            neighbour = (pos[0], pos[1] + j)
            if neighbour[0] < 0 or neighbour[0] >= 120 or neighbour[1] < 0 or neighbour[1] >= 120:
                continue
            if self.world_map[neighbour[0]][neighbour[1]] == 1:
                continue
            neighbours.append(neighbour)
        return neighbours

    def get_path(self):
        return self.path

    def get_euclidean_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def init_h(self):
        for i in range(self.map_x):
            for j in range(self.map_y):
                if self.world_map[i][j] == 1:
                    continue
                # manhattan distance
                # self.h[(i, j)] = self.get_manhattan_distance((i, j), self.goal_pos)
                # euclidean distance
                self.h[(i, j)] = self.get_euclidean_distance((i, j), self.goal_pos)


    def A_star(self):
        while (len(self.open_list)) > 0:
            n = None  # current node
            for pos in self.open_list:
                if n is None or self.g[pos] + self.h[pos] < self.g[n] + self.h[n]:
                    n = pos

            if n is None:
                print('no path found')
                return False

            if n == self.goal_pos:
                self.path = []
                while self.parent_nodes[n] != self.start_pos:  # until reach the start position
                    self.path.append(n)
                    n = self.parent_nodes[n]
                self.path.append(self.start_pos)
                self.path.reverse()
                return True

            neighbours = self.get_4_neighbours(n)
            for neighbour in neighbours:
                if neighbour not in self.open_list and neighbour not in self.closed_list:
                    self.open_list.append(neighbour)
                    self.parent_nodes[neighbour] = n
                    self.g[neighbour] = self.g[n] + self.get_euclidean_distance(n, neighbour)
                else:
                    if self.g[neighbour] > self.g[n] + self.get_euclidean_distance(n, neighbour):
                        self.g[neighbour] = self.g[n] + self.get_euclidean_distance(n, neighbour)
                        self.parent_nodes[neighbour] = n
                        if neighbour in self.closed_list:
                            self.closed_list.remove(neighbour)
                            self.open_list.append(neighbour)

            self.open_list.remove(n)
            self.closed_list.append(n)


###  END CODE HERE  ###


def A_star(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using A* algorithm.

    Arguments:
    world_map -- A 120*120 array indicating map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the start position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    a_star_map = A_Star_Map(world_map, start_pos, goal_pos)
    if a_star_map.A_star():
        path = a_star_map.get_path()
    else:
        raise Exception('no path found')

    ###  END CODE HERE  ###
    return path


if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position of the exploration
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = A_star(map, start_pos, goal_pos)

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
