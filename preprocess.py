import numpy as np
from pymatreader import read_mat
from collections import namedtuple
from dataclasses import dataclass
from numpy import sin, cos, arctan2


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


data = read_mat('./data/SCENE_1.mat')
scene = data['SimulationScenario']


# SensorMeasurements
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. There are  14 sensors , 6 radars and 8 camera.
# 2. `
# nRadars = 6; nCameras = 8;
# nSensors = nRadars + nCameras;
# nObjects = int16(200);
# sensorIDs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
# radarIDs = [9,10,11,12,13,14];
# cameraIDs = [1,2,3,4,5,6,7,8];
# `
# 3. So each sensor will have the total timestamp of 401 measurements for each of its variable
# 4. And there are a total of 14 sensors so it will be a list of 14 elements
# 5. We will store the sensor measurements in a named tuple and push each such a tuple to a list .
# 6. So when  we we need a measurement from one such sensor we can index to the list and get the right value ` sensorID time px py vx vy classID SNR ErrCov`


sensorMeasurements = scene['SensorMeasurements']

sensorTime = sensorMeasurements['Time']
sensorMeasurements = sensorMeasurements['Data']


class CSensorData:
    def __init__(self, px, py, vx, vy, MeasNoiseCov, snr, detTimeStamp, classID):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.MeasNoiseCov = MeasNoiseCov
        self.snr = snr
        self.detTimeStamp = detTimeStamp
        self.objClassID = classID


detectionData = np.empty((14, 1), dtype='object')
for i in range(14):
    detectionData[i] = CSensorData(sensorMeasurements['px'][i], sensorMeasurements['py'][i], sensorMeasurements['vx'][i], sensorMeasurements['vy'][i],
                                   sensorMeasurements['ErrCov'][i], sensorMeasurements['SNR'][i], sensorMeasurements['time'][i], sensorMeasurements['classID'][i])


# LineSensor Data
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


laneData = scene['LaneSensorMeasurements']
laneTime = laneData['Time']
laneData = laneData['Data']


# we will create an array of size 8 and each will have the the required infos
lineSensorData = np.empty((8, 1), dtype=object)


class CLineData:
    def __init__(self, Curvature, CurvRate, CurveLength, HeadingAngle,
                 LateralOffset, XMin, XMax, Width, Time):
        self.Curvature = Curvature
        self.CurvRate = CurvRate
        self.CurveLength = CurveLength
        self.HeadingAngle = HeadingAngle
        self.LateralOffset = LateralOffset
        self.XMin = XMin
        self.XMax = XMax
        self.Width = Width
        self.Time = Time


for i in range(8):
    lineSensorData[i] = CLineData(laneData['Curvature'][i], laneData['CurvRate'][i], laneData['CurveLength'][i], laneData['HeadingAngle'][i],
                                  laneData['LateralOffset'][i], laneData['XMin'][i], laneData['XMax'][i], laneData['Width'][i], laneData['Time'][i])


# EGO DATA
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


egoData = scene['EgoSensorMeasurements']
egoTime = egoData['Time']
egoData = egoData['Data']

egoSensorData = namedtuple(
    'egoSensorData', 'detTimeStamp px  py vx vy yaw yawRate')
egoData = egoSensorData(egoTime, egoData['px'], egoData['py'],
                        egoData['vx'], egoData['vy'], egoData['yaw'], egoData['yawRate'])


# LANE DATA
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1.    `nLaneLines = 2; nLaneDimension = 8; nLaneSensors = 8;`
# 2. We will get lane sensor data for all the 8 camera sensors we are having but we need only two(I think they are probably the front ones)


laneData = scene['LaneSensorMeasurements']
laneTime = laneData['Time']
laneData = laneData['Data']


class ClaneSensorData:
    def __init__(self, LateralOffset, HeadingAngle, Curvature, CurvatureDerivative, CurveLength,
                 LineWidth, MaximumValidX, MinimumValidX, detTimeStamp):
        self.LateralOffset = LateralOffset
        self.HeadingAngle = HeadingAngle
        self.Curvature = Curvature
        self.CurvatureDerivative = CurvatureDerivative
        self.CurveLength = CurveLength
        self.LineWidth = LineWidth
        self.MaximumValidX = MaximumValidX
        self.MinimumValidX = MinimumValidX
        self.detTimeStamp = detTimeStamp


laneSensorData = np.empty((2, 1), dtype=object)


for i in range(2):
    laneSensorData[i] = ClaneSensorData(laneData['LateralOffset'][i], laneData['HeadingAngle'][i], laneData['Curvature'][i], laneData['CurvRate'][i],
                                        laneData['CurveLength'][i], laneData['Width'][i], laneData['XMax'][i], laneData['XMin'][i], laneData['Time'][i])
