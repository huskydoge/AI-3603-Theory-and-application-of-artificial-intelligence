import math

import numpy as np

from .car import Car


class Cost:
    reverse = np.inf
    directionChange = 250
    steerAngle = 1
    steerAngleChange = 500000
    hybridWeight = 50

    # reverse = np.inf
    # directionChange = 250
    # steerAngle = 1
    # steerAngleChange = 500
    # hybridWeight = 50


def reedsSheppCost(currentNode, path):
    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * Cost.reverse  # reverse cost, punish the behavior of reverse

    # Direction change cost
    for i in range(len(path.lengths) - 1):
        if path.lengths[i] * path.lengths[i + 1] < 0:
            cost += Cost.directionChange

    # Steering Angle Cost
    for i in path.ctypes:
        # Check types which are not straight line
        if i != "start":
            cost += Car.maxSteerAngle * Cost.steerAngle

    # Steering Angle change cost
    turnAngle = [0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turnAngle[i] = - Car.maxSteerAngle
        if path.ctypes[i] == "WB":
            turnAngle[i] = Car.maxSteerAngle

    for i in range(len(path.lengths) - 1):
        cost += abs(turnAngle[i + 1] - turnAngle[i]) * Cost.steerAngleChange

    return cost


def simulatedPathCost(currentNode, moves, simulationLength):
    # Previos Node Cost
    cost = currentNode.cost

    # Distance cost
    if moves[1] == 1:
        cost += simulationLength
    else:
        cost += simulationLength * Cost.reverse

    # Direction change cost
    if currentNode.direction != moves[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += moves[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(moves[0] - currentNode.steeringAngle) * Cost.steerAngleChange

    return cost


def getEuclideanCost(holonomicMove):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMove[0], holonomicMove[1])
