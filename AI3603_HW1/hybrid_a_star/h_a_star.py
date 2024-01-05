import heapq
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from heapdict import heapdict

from .car import drawCar, getCarMoves, getHolonomicMove
from .costSetting import Cost, getEuclideanCost
from .map import obstaclesMap, calculateMapParameters, getMapFromLocal
from .node import Node, HolonomicNode, index, getHolonomicNodeIndex, reedsSheppNode, holonomicNodeIsValid, \
    simulationNode

show_animation = False

obstacleX, obstacleY = getMapFromLocal()


def init_h(goalNode, mapParameters):
    print("Calculating Holonomic Cost......")
    pos = [round(goalNode.traj[-1][0] / mapParameters.xyResolution),
           round(goalNode.traj[-1][1] / mapParameters.xyResolution)]

    gNode = HolonomicNode(pos, 0, tuple(pos))

    # obs
    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)

    holonomicMove = getHolonomicMove()

    openList = {getHolonomicNodeIndex(gNode): gNode}
    closedList = {}

    priorityQueue = []
    heapq.heappush(priorityQueue, (gNode.cost, getHolonomicNodeIndex(gNode)))

    while True:
        if not openList:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openList[currentNodeIndex]

        if show_animation:  # pragma: no cover
            # print("show holonomic cost calculation......")
            plt.plot(currentNode.pos[0], currentNode.pos[1], ".goal")
            plt.title("calculate distance heuristic")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        openList.pop(currentNodeIndex)
        closedList[currentNodeIndex] = currentNode

        for i in range(len(holonomicMove)):
            neighbourNode = HolonomicNode([currentNode.pos[0] + holonomicMove[i][0],
                                           currentNode.pos[1] + holonomicMove[i][1]],
                                          currentNode.cost + getEuclideanCost(holonomicMove[i]), currentNodeIndex)
            # check whether the node is in bound and not obstacles
            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue

            neighbourNodeIndex = getHolonomicNodeIndex(neighbourNode)

            if neighbourNodeIndex not in closedList:
                if neighbourNodeIndex in openList:
                    if neighbourNode.cost < openList[neighbourNodeIndex].cost:
                        openList[neighbourNodeIndex].cost = neighbourNode.cost
                        openList[neighbourNodeIndex].parentPos = neighbourNode.parentPos
                else:
                    openList[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

    holonomicCost = [[np.inf for i in range(max(mapParameters.obstacleY))] for i in range(max(mapParameters.obstacleX))]

    for nodes in closedList.values():
        holonomicCost[nodes.pos[0]][nodes.pos[1]] = nodes.cost

    print("Holonomic Cost Calculated")
    return holonomicCost


def backtrack(startNode, goalNode, closedList, plt):
    # Goal Node data
    startNodeIndex = index(startNode)
    currentNodeIndex = goalNode.parentPos
    currentNode = closedList[currentNodeIndex]
    x = []
    y = []
    yaw = []

    # Iterate till we reach start node from goal node
    while currentNodeIndex != startNodeIndex:
        a, b, c = zip(*currentNode.traj)
        x += a[::-1]
        y += b[::-1]
        yaw += c[::-1]
        currentNodeIndex = currentNode.parentPos
        currentNode = closedList[currentNodeIndex]
    return x[::-1], y[::-1], yaw[::-1]


def run_A_Star(start, goal, mapParameters, plt=None):
    # Compute position for start and Goal node, should be integer ! Proper Resolution could get algorithm run faster
    start_pos = [round(start[0] / mapParameters.xyResolution), round(start[1] / mapParameters.xyResolution),
                 round(start[2] / mapParameters.yawResolution)]
    goal_pos = [round(goal[0] / mapParameters.xyResolution),
                round(goal[1] / mapParameters.xyResolution),
                round(goal[2] / mapParameters.yawResolution)]

    # get possible car moves
    moves = getCarMoves()

    # start and goal Node
    startNode = Node(start_pos, [start], 0, 1, 0, parentPos=tuple(start_pos))
    goalNode = Node(goal_pos, [goal], 0, 1, 0, parentPos=tuple(goal_pos))

    # get Heuristric Map, h[pos[0]][pos[1]] = h(x,y), 2D array, could replace dict(x,y)
    h = init_h(goalNode, mapParameters)

    # Add start node to open Set
    openList = {index(startNode): startNode}
    closedList = {}

    # Create a priority queue for acquiring nodes based on their cost in ascending order
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index(startNode)] = startNode.cost + Cost.hybridWeight * h[startNode.pos[0]][
        startNode.pos[1]]
    counter = 0

    # Run loop while path is found or openList is empty
    while True:
        counter += 1
        # if empty then no possible solution, return None
        if not openList:
            return None

        # Pop first node in the priority queue, the one with lowest cost
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openList[currentNodeIndex]

        # Move current Node from openList to closedList
        openList.pop(currentNodeIndex)
        closedList[currentNodeIndex] = currentNode

        # Get Reed-Shepp Node if available
        rSNode = reedsSheppNode(currentNode, goalNode, mapParameters)

        # Id Reeds-Shepp Path is found exit
        if rSNode:
            closedList[index(rSNode)] = rSNode
            print("Path Found with Reeds-Shepp")
            break

        # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
        if currentNodeIndex == index(goalNode):
            print("Path Found")
            # print(currentNode.traj[-1])
            break

        # Get all simulated Nodes from current node, move = [steeringAngle, direction]
        for i in range(len(moves)):
            simulatedNode = simulationNode(currentNode, moves[i], mapParameters)

            # Check if path is within map and no collision occurs
            if not simulatedNode:
                continue

            # Draw Simulated Node
            x, y, z = zip(*simulatedNode.traj)
            # plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            if simulatedNodeIndex not in closedList:

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulatedNodeIndex not in openList:
                    openList[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = (simulatedNode.cost + Cost.hybridWeight *
                                                     h[simulatedNode.pos[0]][
                                                         simulatedNode.pos[1]])
                else:
                    if simulatedNode.cost < openList[simulatedNodeIndex].cost:
                        openList[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = (simulatedNode.cost + Cost.hybridWeight *
                                                         h[simulatedNode.pos[0]][
                                                             simulatedNode.pos[1]])

    # Backtrack
    x, y, yaw = backtrack(startNode, goalNode, closedList, plt)
    print("Path Length: ", len(x))
    print("Number of Nodes Expanded: ", len(closedList))
    print("Number of Nodes Visited: ", len(openList) + len(closedList))
    print("Counter: ", counter)
    return x, y, yaw


def main():
    # Set Start, Goal x, y, yaw
    start = [10, 10, np.deg2rad(0)]
    goal = [100, 100, np.deg2rad(180)]

    # Get Obstacle Map
    obstacleX, obstacleY = getMapFromLocal()

    print("Getting Map.......")

    plt.plot(obstacleX, obstacleY, ".k")
    plt.arrow(start[0], start[1], 2 * math.cos(start[2]), 2 * math.sin(start[2]), width=0.5)
    plt.arrow(goal[0], goal[1], 2 * math.cos(goal[2]), 2 * math.sin(goal[2]), width=0.5)
    plt.grid(True)
    plt.axis("equal")

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))

    print("Start Planning.......")
    # Run Hybrid A*
    x, y, yaw = run_A_Star(start, goal, mapParameters, plt)

    # Draw Map and Path

    # plt.xlim(min(obstacleX), max(obstacleX))
    # plt.ylim(min(obstacleY), max(obstacleY))
    plt.plot(obstacleX, obstacleY, "sk")
    plt.plot(x, y, linewidth=2, color='r', zorder=0)
    # plt.title("Hybrid A*")

    # Draw Car path
    # plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
    # plt.plot(obstacleX, obstacleY, "sk")
    # for k in np.arange(0, len(x), 2):
    #     plt.xlim(min(obstacleX), max(obstacleX))
    #     plt.ylim(min(obstacleY), max(obstacleY))
    #     drawCar(x[k], y[k], yaw[k])
    #     plt.arrow(x[k], y[k], 1 * math.cos(yaw[k]), 1 * math.sin(yaw[k]), width=0.1)
    #     plt.title("Hybrid A*")
    #
    plt.plot(start[0], start[1], "xr")
    plt.plot(goal[0], goal[1], "xb")
    # plt.show()

    # exit(0)
    data_ = zip(x, y, yaw)
    data = [i for i in data_]
    fig = plt.figure()
    # Draw Animated Car
    animator = animation.FuncAnimation(fig, plot_car_animation, frames=data, interval=80)
    animator.save("task3.gif", writer='pillow')
    plt.show()


def plot_car_animation(data):
    x, y, yaw = data
    plt.cla()
    plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
    print(x, y, yaw)
    plt.xlim(min(obstacleX), max(obstacleX))
    plt.ylim(min(obstacleY), max(obstacleY))
    plt.plot(obstacleX, obstacleY, "sk")
    drawCar(x, y, yaw)
    plt.arrow(x, y, 1 * math.cos(yaw), 1 * math.sin(yaw), width=.1)
    plt.grid(True)
    plt.axis("equal")
    # plt.title("Hybrid A*")
    # plt.pause(0.001)


if __name__ == '__main__':
    main()
