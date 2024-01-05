import math

import numpy as np
from matplotlib import pyplot as plt


class Car:
    maxSteerAngle = 0.4  # max
    steerPrecision = 10  # to cut down on the number of steering angles, faster but not precise
    wheelBase = 3.5
    axleToFront = 4.5
    axleToRear = 1
    width = 3


def drawCar(x, y, yaw, color='black'):
    car = np.array([[-Car.axleToRear, -Car.axleToRear, Car.axleToFront, Car.axleToFront, -Car.axleToRear],
                    [Car.width / 2, -Car.width / 2, -Car.width / 2, Car.width / 2, Car.width / 2]])

    rotationZ = np.array([[math.cos(yaw), -math.sin(yaw)],
                          [math.sin(yaw), math.cos(yaw)]])
    car = np.dot(rotationZ, car)
    car += np.array([[x], [y]])
    plt.plot(car[0, :], car[1, :], color)


def getCarMoves():
    # Possible movement for Car
    direction = 1
    moves = []
    for i in np.arange(Car.maxSteerAngle, -(Car.maxSteerAngle + Car.maxSteerAngle / Car.steerPrecision),
                       -Car.maxSteerAngle / Car.steerPrecision):
        moves.append([i, direction])
        moves.append([i, -direction])
    return moves


def getHolonomicMove():
    # Action set for a Holonomic Robot (8-Directions, like in task 2)
    holonomicMove = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMove
