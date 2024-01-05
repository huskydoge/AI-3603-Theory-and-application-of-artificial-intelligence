import math
import numpy as np
import scipy.spatial.kdtree as kd


class MapParameters:
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX,
                 obstacleY):
        self.mapMinX = mapMinX  # map min x coordinate(0)
        self.mapMinY = mapMinY  # map min y coordinate(0)
        self.mapMaxX = mapMaxX  # map max x coordinate
        self.mapMaxY = mapMaxY  # map max y coordinate
        self.xyResolution = xyResolution  # grid block length
        self.yawResolution = yawResolution  # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree  # KDTree representating obstacles
        self.obstacleX = obstacleX  # Obstacle x coordinate list
        self.obstacleY = obstacleY  # Obstacle y coordinate list


def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
    # calculate min max map grid index based on obstacles in map
    mapMinX = round(min(obstacleX) / xyResolution)
    mapMinY = round(min(obstacleY) / xyResolution)
    mapMaxX = round(max(obstacleX) / xyResolution)
    mapMaxY = round(max(obstacleY) / xyResolution)

    # create a KDTree to represent obstacles
    ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

    return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX,
                         obstacleY)


def obstaclesMap(obstacleX, obstacleY, xyResolution):
    # Compute Grid Index for obstacles
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]

    # Set all Grid locations to No Obstacle
    obstacles = [[False for i in range(max(obstacleY))] for i in range(max(obstacleX))]

    # Set Grid Locations with obstacles to True
    for x in range(max(obstacleX)):
        for y in range(max(obstacleY)):
            for i, j in zip(obstacleX, obstacleY):
                if math.hypot(i - x, j - y) <= 1 / 2:
                    obstacles[i][j] = True
                    break

    return obstacles


def getMapFromLocal():
    # Build Map
    obstacleX, obstacleY = [], []
    MAP_PATH = '3-map/map.npy'
    map = np.load(MAP_PATH)
    # print(map.shape)
    # load obstacles
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 1:
                obstacleX.append(i)
                obstacleY.append(j)

    return obstacleX, obstacleY
