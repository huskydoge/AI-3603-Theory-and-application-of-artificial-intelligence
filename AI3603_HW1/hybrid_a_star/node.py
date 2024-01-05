import math

from heapdict import heapdict

import CurvesGenerator.reeds_shepp as rsCurve
from .car import Car
from .costSetting import reedsSheppCost, simulatedPathCost


class Node:
    def __init__(self, pos, traj, steeringAngle, direction, cost, parentPos):
        self.pos = pos  # x, y, yaw position
        self.traj = traj  # trajectory x, y  of a simulated node
        self.steeringAngle = steeringAngle  # steering angle throughout the trajectory
        self.direction = direction  # direction throughout the trajectory
        self.cost = cost  # node cost
        self.parentPos = parentPos  # parent node position


class HolonomicNode:
    # Node for Holonomic Model
    def __init__(self, pos, cost, parentPos):
        self.pos = pos
        self.cost = cost
        self.parentPos = parentPos


def index(Node):
    # Index is set to node's position parameters
    return tuple([Node.pos[0], Node.pos[1], Node.pos[2]])


def getHolonomicNodeIndex(HolonomicNode):
    # Index is set to node's position parameters
    return tuple([HolonomicNode.pos[0], HolonomicNode.pos[1]])


def simulationNode(currentNode, moves, mapParameters, simulationLength=4, step=0.8):
    # Simulate node in the next traj using given current Node and Moves
    traj = []
    angle = rsCurve.pi_2_pi(
        currentNode.traj[-1][2] + moves[1] * step / Car.wheelBase * math.tan(moves[0]))
    traj.append([currentNode.traj[-1][0] + moves[1] * step * math.cos(angle),
                 currentNode.traj[-1][1] + moves[1] * step * math.sin(angle),
                 rsCurve.pi_2_pi(angle + moves[1] * step / Car.wheelBase * math.tan(moves[0]))])
    for i in range(int((simulationLength / step)) - 1):
        traj.append([traj[i][0] + moves[1] * step * math.cos(traj[i][2]),
                     traj[i][1] + moves[1] * step * math.sin(traj[i][2]),
                     rsCurve.pi_2_pi(
                         traj[i][2] + moves[1] * step / Car.wheelBase * math.tan(moves[0]))])

    # Find position
    pos = [round(traj[-1][0] / mapParameters.xyResolution),
           round(traj[-1][1] / mapParameters.xyResolution),
           round(traj[-1][2] / mapParameters.yawResolution)]

    # Check if node is valid
    if not isValid(traj, pos, mapParameters):
        return None

    # Calculate Cost of the node
    cost = simulatedPathCost(currentNode, moves, simulationLength)

    return Node(pos, traj, moves[0], moves[1], cost, index(currentNode))


def isValid(traj, pos, mapParameters):
    # Check if Node is out of map bounds
    if pos[0] <= mapParameters.mapMinX or pos[0] >= mapParameters.mapMaxX or \
            pos[1] <= mapParameters.mapMinY or pos[1] >= mapParameters.mapMaxY:
        return False

    # Check if Node is colliding with an obstacle
    if collision(traj, mapParameters):
        return False
    return True


def reedsSheppNode(currentNode, goalNode, mapParameters):
    # Get x, y, yaw of currentNode and goalNode
    startX, startY, startYaw = currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]
    goalX, goalY, goalYaw = goalNode.traj[-1][0], goalNode.traj[-1][1], goalNode.traj[-1][2]

    # Instantaneous Radius of Curvature
    radius = math.tan(Car.maxSteerAngle) / Car.wheelBase

    #  Find all possible reeds-shepp paths between current and goal node
    reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 1)

    # Check if reedsSheppPaths is empty
    if not reedsSheppPaths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    costQueue = heapdict()
    for path in reedsSheppPaths:
        costQueue[path] = reedsSheppCost(currentNode, path)

    # Find first path in priority queue that is collision free
    while len(costQueue) != 0:
        path = costQueue.popitem()[0]
        traj = [[path.x[k], path.y[k], path.yaw[k]] for k in range(len(path.x))]
        if not collision(traj, mapParameters):
            cost = reedsSheppCost(currentNode, path)
            return Node(goalNode.pos, traj, None, None, cost, index(currentNode))

    return None


def collision(traj, mapParameters):
    carRadius = (Car.axleToFront + Car.axleToRear) / 2 + 1
    dl = (Car.axleToFront - Car.axleToRear) / 2
    for i in traj:
        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        pointsInObstacle = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], carRadius + 1)
        if not pointsInObstacle:
            continue

        for p in pointsInObstacle:
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < carRadius + 1 or abs(dy) < Car.width / 2 + 1:
                return True

    return False


def holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
    # Check if Node is out of map bounds
    if neighbourNode.pos[0] <= mapParameters.mapMinX or \
            neighbourNode.pos[0] >= mapParameters.mapMaxX or \
            neighbourNode.pos[1] <= mapParameters.mapMinY or \
            neighbourNode.pos[1] >= mapParameters.mapMaxY:
        return False

    # Check if Node on obstacle
    if obstacles[neighbourNode.pos[0]][neighbourNode.pos[1]]:
        return False

    return True


def checkNearObs(x, y, ox, oy, radius):
    for iox, ioy in zip(ox, oy):
        d = math.sqrt((iox - x) ** 2 + (ioy - y) ** 2)
        if d <= radius:
            return True  # collision
    return False
