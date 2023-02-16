import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from numpy import sin, cos, arctan2


def setRadarIntrinsicParam(nRadarTypes,
                           RADAR_MAX_RANGE,
                           RADAR_MAX_AZIMUTH,
                           RADAR_MAX_ELEVATION,
                           RADAR_MAX_RANGE_RATE,
                           RADAR_RANGE_RES,
                           RADAR_AZIMUTH_RES,
                           RADAR_ELEVATION_RES,
                           RADAR_RANGE_RATE_RES,
                           RADAR_RANGE_ERR_VAR,
                           RADAR_AZIMUTH_ERR_VAR,
                           RADAR_ELEVATION_ERR_VAR,
                           RADAR_RANGE_RATE_ERR_VAR,
                           RADAR_PD,
                           RADAR_FA,
                           RADAR_FOV_BOUNDARY_PTS_RANGE,
                           RADAR_FOV_BOUNDARY_PTS_AZIMUTH,
                           RadarINTRINSICparam):

    for i in range(nRadarTypes):
        RadarINTRINSICparam[i, 0].RadarType = i
        RadarINTRINSICparam[i, 0].MaxRange = RADAR_MAX_RANGE[i]
        RadarINTRINSICparam[i, 0].MaxAzimuth = RADAR_MAX_AZIMUTH[i]
        RadarINTRINSICparam[i, 0].MaxElevation = RADAR_MAX_ELEVATION[i]
        RadarINTRINSICparam[i, 0].MaxRangeRate = RADAR_MAX_RANGE_RATE[i]
        RadarINTRINSICparam[i, 0].RangeResolution = RADAR_RANGE_RES[i]
        RadarINTRINSICparam[i, 0].AzimuthResolution = RADAR_AZIMUTH_RES[i]
        RadarINTRINSICparam[i, 0].ElevationResolution = RADAR_ELEVATION_RES[i]
        RadarINTRINSICparam[i, 0].RangeRateResolution = RADAR_RANGE_RATE_RES[i]
        RadarINTRINSICparam[i, 0].RangeErrVariance = RADAR_RANGE_ERR_VAR[i]
        RadarINTRINSICparam[i, 0].AzimuthErrVariance = RADAR_AZIMUTH_ERR_VAR[i]
        RadarINTRINSICparam[i,
                            0].ElevationErrVariance = RADAR_ELEVATION_ERR_VAR[i]
        RadarINTRINSICparam[i,
                            0].RangeRateErrVariance = RADAR_RANGE_RATE_ERR_VAR[i]
        RadarINTRINSICparam[i, 0].ProbOfDetection = RADAR_PD[i]
        RadarINTRINSICparam[i, 0].FalseAlarmRate = RADAR_FA[i]
        RadarINTRINSICparam[i,
                            0].FOVRangePoints = RADAR_FOV_BOUNDARY_PTS_RANGE[i]
        RadarINTRINSICparam[i,
                            0].FOVAzimuthPts = RADAR_FOV_BOUNDARY_PTS_AZIMUTH[i]

    return RadarINTRINSICparam


def setSensorExtrinsicParam(nSensors,
                            SENSOR_TYPE,
                            IS_SENSOR_ACTIVE,
                            SENSOR_MOUNT_X,
                            SENSOR_MOUNT_Y,
                            SENSOR_MOUNT_Z,
                            SENSOR_MOUNT_ROLL,
                            SENSOR_MOUNT_PITCH,
                            SENSOR_MOUNT_YAW,
                            SENSOR_nMEAS,
                            SensorEXTRINSICparam, ROTATION_CONV):

    for i in range(nSensors):
        DEG2RAD = np.pi/180
        SensorEXTRINSICparam[i, 0].SensorID = i
        SensorEXTRINSICparam[i, 0].SensorType = SENSOR_TYPE[i]
        SensorEXTRINSICparam[i, 0].isActive = IS_SENSOR_ACTIVE[i]
        SensorEXTRINSICparam[i, 0].MountX = SENSOR_MOUNT_X[i]
        SensorEXTRINSICparam[i, 0].MountY = SENSOR_MOUNT_Y[i]
        SensorEXTRINSICparam[i, 0].MountZ = SENSOR_MOUNT_Z[i]
        SensorEXTRINSICparam[i, 0].MountYaw = SENSOR_MOUNT_YAW[i]
        SensorEXTRINSICparam[i, 0].MountPitch = SENSOR_MOUNT_PITCH[i]
        SensorEXTRINSICparam[i, 0].MountRoll = SENSOR_MOUNT_ROLL[i]
        SensorEXTRINSICparam[i, 0].nMeas = SENSOR_nMEAS[i]
        SensorEXTRINSICparam[i, 0].RotMat2D = np.array([[cos(SENSOR_MOUNT_YAW[i]*DEG2RAD),             -sin(
            ROTATION_CONV*SENSOR_MOUNT_YAW[i]*DEG2RAD)], [sin(ROTATION_CONV*SENSOR_MOUNT_YAW[i]*DEG2RAD), cos(SENSOR_MOUNT_YAW[i]*DEG2RAD)]])

        SensorEXTRINSICparam[i, 0].TranslationVec = np.array(
            [[SENSOR_MOUNT_X[i]], [SENSOR_MOUNT_Y[i]]])

    return SensorEXTRINSICparam


def setCameraIntrinsicParam(nCameraTypes,
                            CAMERA_MAX_RANGE, CAMERA_MAX_AZIMUTH, CAMERA_MAX_ELEVATION,
                            CAMERA_LONG_ERR_VAR, CAMERA_LAT_ERR_VAR,
                            CAMERA_PD, CAMERA_FA,
                            CAMERA_FOV_BOUNDARY_PTS_RANGE, CAMERA_FOV_BOUNDARY_PTS_AZIMUTH,
                            CameraINTRINSICparam):
    for i in range(nCameraTypes):
        CameraINTRINSICparam[i, 0].CameraType = i
        CameraINTRINSICparam[i, 0].RectificationMatrix = np.zeros(
            (3, 3), dtype=float)
        CameraINTRINSICparam[i, 0].ProjectionMatrix = np.zeros(
            (3, 3), dtype=float)
        CameraINTRINSICparam[i, 0].MaxRange = CAMERA_MAX_RANGE[i]
        CameraINTRINSICparam[i, 0].MaxAzimuth = CAMERA_MAX_AZIMUTH[i]
        CameraINTRINSICparam[i, 0].MaxElevation = CAMERA_MAX_ELEVATION[i]
        CameraINTRINSICparam[i,
                             0].LongitudinalErrVariance = CAMERA_LONG_ERR_VAR[i]
        CameraINTRINSICparam[i, 0].LateralErrVariance = CAMERA_LAT_ERR_VAR[i]
        CameraINTRINSICparam[i, 0].ProbOfDetection = CAMERA_PD[i]
        CameraINTRINSICparam[i, 0].FalseAlarmRate = CAMERA_FA[i]
        CameraINTRINSICparam[i,
                             0].FOVRangePoints = CAMERA_FOV_BOUNDARY_PTS_RANGE[i]
        CameraINTRINSICparam[i,
                             0].FOVAzimuthPts = CAMERA_FOV_BOUNDARY_PTS_AZIMUTH[i]

    return CameraINTRINSICparam


def isMeasurementValid(px, py):
    if((abs(px) + abs(py)) >= 0.000001):
        isValid = True
    else:
        isValid = False

    return isValid


def set_RADAR_SENSOR_MEAS_DATA(RAD_Sensor_Simulated_Data, timeIdx, snsrIdx, nMeas, RADAR_CAN_BUS):
    # Set Radar Sensor Data with the Array of Structure
    # INPUT : RAD_Sensor_Simulated_Data : a structure of 2D arrays of simulated sensor data
    #         timeIdx : time index
    #         snsrIdx : radar sensor index
    #         nMeas :  maximum number of measurements possible
    #         RADAR_CAN_BUS : initialized array of structure of radar sensor measurements
    # OUTPUT : RADAR_CAN_BUS : array of structure of radar sensor measurements with the following fields
    #        : px, py : measurements
    #        : measNoise : measurement noise covariance
    #        : measID : valid meas id
    #        : sensorID : sensor ID that gave the detection
    #        : detTimeStamp : detected time of the measurement
    #        : snr : signal to noise ratio (applicable only for radar)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nValidMeas = 0
    # %MeasNoiseCov = single([0.3, 0; ...   % (px, py)
    # %                       0, 0.6]);
    MeasNoiseCov = np.array([[2, 0], [0, 3.3]])

    for objIdx in range(nMeas):
        if (isMeasurementValid(RAD_Sensor_Simulated_Data.px[timeIdx, objIdx], RAD_Sensor_Simulated_Data.px[timeIdx, objIdx])):
            nValidMeas = nValidMeas + 1
            RADAR_CAN_BUS[snsrIdx, objIdx].px = float(
                RAD_Sensor_Simulated_Data.px[timeIdx, objIdx])
            RADAR_CAN_BUS[snsrIdx, objIdx].py = float(
                RAD_Sensor_Simulated_Data.py[timeIdx, objIdx])
            azimuth = arctan2(
                RADAR_CAN_BUS[snsrIdx, objIdx].py, RADAR_CAN_BUS[snsrIdx, objIdx].px)
            Rotation = np.array([[cos(azimuth), -sin(azimuth)],
                                [sin(azimuth), cos(azimuth)]])
            MeasNoiseCov = Rotation * MeasNoiseCov * Rotation.transpose()
            RADAR_CAN_BUS[snsrIdx, objIdx].measNoise = MeasNoiseCov
            RADAR_CAN_BUS[snsrIdx, objIdx].measID = nValidMeas
            RADAR_CAN_BUS[snsrIdx, objIdx].sensorID = snsrIdx
            RADAR_CAN_BUS[snsrIdx, objIdx].detTimeStamp = float(
                RAD_Sensor_Simulated_Data.detTimeStamp[timeIdx])
            RADAR_CAN_BUS[snsrIdx, objIdx].snr = float(
                RAD_Sensor_Simulated_Data.snr[timeIdx, objIdx])

        else:
            RADAR_CAN_BUS[snsrIdx, objIdx].px = 0.0
            RADAR_CAN_BUS[snsrIdx, objIdx].py = 0.0
            RADAR_CAN_BUS[snsrIdx, objIdx].measNoise = np.zeros(
                MeasNoiseCov.shape)
            RADAR_CAN_BUS[snsrIdx, objIdx].measID = 0
            RADAR_CAN_BUS[snsrIdx, objIdx].sensorID = snsrIdx
            RADAR_CAN_BUS[snsrIdx, objIdx].detTimeStamp = 0.0
            RADAR_CAN_BUS[snsrIdx, objIdx].snr = 0.0

    return RADAR_CAN_BUS


def RAD_SENSOR_INTERFACE(RAD1_Sensor_Simulated_Data,
                         RAD2_Sensor_Simulated_Data,
                         RAD3_Sensor_Simulated_Data,
                         RAD4_Sensor_Simulated_Data,
                         RAD5_Sensor_Simulated_Data,
                         RAD6_Sensor_Simulated_Data,
                         RADAR_CAN_BUS, t, nRadars, nMeas):
    # Interface Radar Sensor Data with the Array of Structure
    # INPUT  : RAD1_Sensor_Simulated_Data : Radar 1 simulated measurements
    #          RAD2_Sensor_Simulated_Data : Radar 2 simulated measurements
    #          RAD3_Sensor_Simulated_Data : Radar 3 simulated measurements
    #          RAD4_Sensor_Simulated_Data : Radar 4 simulated measurements
    #          RAD5_Sensor_Simulated_Data : Radar 5 simulated measurements
    #          RAD6_Sensor_Simulated_Data : Radar 6 simulated measurements
    #          RADAR_CAN_BUS : init array of structure of radar measurements
    #          t : time index
    #          nRadars : number of radars
    #          nMeas : number of measurements
    # OUTPUT : RADAR_CAN_BUS : array of structure of radar measurements
    for snsrIdx in range(nRadars):
        if(snsrIdx == 0):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD1_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        elif(snsrIdx == 1):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD2_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        elif(snsrIdx == 2):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD3_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        elif(snsrIdx == 3):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD4_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        elif(snsrIdx == 4):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD5_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        elif(snsrIdx == 5):
            RADAR_CAN_BUS = set_RADAR_SENSOR_MEAS_DATA(
                RAD6_Sensor_Simulated_Data, t, snsrIdx, nMeas, RADAR_CAN_BUS)
        else:
            print(snsrIdx)
            print("number of radars exceeds the upper limit")

    return RADAR_CAN_BUS


def set_CAMERA_SENSOR_MEAS_DATA(CAM_Sensor_Simulated_Data, timeIdx, snsrIdx, nMeas, CAMERA_CAN_BUS):
    # Set CAMERA Sensor Data with the Array of Structure
    # INPUT : CAM_Sensor_Simulated_Data : a structure of 2D arrays of simulated sensor data
    #         timeIdx : time index
    #         snsrIdx : camera sensor index
    #         nMeas :  maximum number of measurements possible
    #         CAMERA_CAN_BUS : initialized array of structure of camera sensor measurements
    # OUTPUT : CAMERA_CAN_BUS : array of structure of camera sensor measurements with the following fields
    #        : px, py, vx, vy : measurements
    #        : measNoise : measurement noise covariance
    #        : measID : valid meas id
    #        : sensorID : sensor ID that gave the detection
    #        : detTimeStamp : detected time of the measurement
    #        : objClassID : object classification (applicable only for camera)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nValidMeas = 0
    MeasNoiseCov = np.array([[3.7, 0],
                             [0, 2.1]])
    for objIdx in range(nMeas):
        if(isMeasurementValid(CAM_Sensor_Simulated_Data.px[timeIdx, objIdx], CAM_Sensor_Simulated_Data.py[timeIdx, objIdx])):
            nValidMeas = nValidMeas + 1
            CAMERA_CAN_BUS[snsrIdx,
                           objIdx].px = CAM_Sensor_Simulated_Data.px[timeIdx, objIdx]
            CAMERA_CAN_BUS[snsrIdx,
                           objIdx].py = CAM_Sensor_Simulated_Data.py[timeIdx, objIdx]
            CAMERA_CAN_BUS[snsrIdx, objIdx].measNoise = MeasNoiseCov
            CAMERA_CAN_BUS[snsrIdx, objIdx].measID = nValidMeas
            CAMERA_CAN_BUS[snsrIdx, objIdx].sensorID = snsrIdx
            CAMERA_CAN_BUS[snsrIdx,
                           objIdx].detTimeStamp = CAM_Sensor_Simulated_Data.detTimeStamp[timeIdx]
            CAMERA_CAN_BUS[snsrIdx,
                           objIdx].objClassID = CAM_Sensor_Simulated_Data.objClassID[timeIdx, objIdx]
        else:
            CAMERA_CAN_BUS[snsrIdx, objIdx].px = 0.0
            CAMERA_CAN_BUS[snsrIdx, objIdx].py = 0.0
            CAMERA_CAN_BUS[snsrIdx, objIdx].measNoise = np.zeros(
                (MeasNoiseCov.shape), dtype=float)
            CAMERA_CAN_BUS[snsrIdx, objIdx].measID = 0
            CAMERA_CAN_BUS[snsrIdx, objIdx].sensorID = snsrIdx
            CAMERA_CAN_BUS[snsrIdx, objIdx].detTimeStamp = 0.0
            CAMERA_CAN_BUS[snsrIdx, objIdx].objClassID = 0.0

    return CAMERA_CAN_BUS


def CAM_SENSOR_INTERFACE(CAM1_Sensor_Simulated_Data,
                         CAM2_Sensor_Simulated_Data,
                         CAM3_Sensor_Simulated_Data,
                         CAM4_Sensor_Simulated_Data,
                         CAM5_Sensor_Simulated_Data,
                         CAM6_Sensor_Simulated_Data,
                         CAM7_Sensor_Simulated_Data,
                         CAM8_Sensor_Simulated_Data,
                         CAMERA_CAN_BUS, t, nCameras, nMeas):

    # Interface Camera Sensor Data with the Array of Structure
    # INPUT  : CAM1_Sensor_Simulated_Data : Camera 1 simulated measurements
    #          CAM2_Sensor_Simulated_Data : Camera 2 simulated measurements
    #          CAM3_Sensor_Simulated_Data : Camera 3 simulated measurements
    #          CAM4_Sensor_Simulated_Data : Camera 4 simulated measurements
    #          CAM5_Sensor_Simulated_Data : Camera 5 simulated measurements
    #          CAM6_Sensor_Simulated_Data : Camera 6 simulated measurements
    #          CAM7_Sensor_Simulated_Data : Camera 7 simulated measurements
    #          CAM8_Sensor_Simulated_Data : Camera 8 simulated measurements
    #          CAMERA_CAN_BUS : init array of structure of camera measurements
    #          t : time index
    #          nCameras : number of cameras
    #          nMeas : number of measurements
    # OUTPUT : CAMERA_CAN_BUS : array of structure of camera measurements
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    for snsrIdx in range(nCameras):
        if(snsrIdx == 0):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM1_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 1):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM2_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 2):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM3_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 3):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM4_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 4):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM5_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 5):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM6_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 6):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM7_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        elif (snsrIdx == 7):
            CAMERA_CAN_BUS = set_CAMERA_SENSOR_MEAS_DATA(
                CAM8_Sensor_Simulated_Data, t, snsrIdx, nMeas, CAMERA_CAN_BUS)

        else:
            print('Sensor out of range')

        return CAMERA_CAN_BUS

# note the egoCanBus and EgoSensorSimulatedData are not numpy arrays, they are named tuples


def EGO_SENSOR_INTERFACE(EGO_Sensor_Simulated_Data, EGO_CAN_BUS, t):
    # Interface EGO Sensor Data with structure
    # INPUT  : EGO_Sensor_Simulated_Data : Simulated measurements of the ego sensors
    #        : EGO_CAN_BUS : init structure of ego sensor measurements having the following parameters
    #        : t : time index
    # OUTPUT : EGO_CAN_BUS : structure of ego sensor measurements having the following parameters :
    #        : px, py, vx, vy, yaw, yawRate
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    EGO_CAN_BUS.px = EGO_Sensor_Simulated_Data.px[t]
    EGO_CAN_BUS.py = EGO_Sensor_Simulated_Data.py[t]
    EGO_CAN_BUS.vx = EGO_Sensor_Simulated_Data.vx[t]
    EGO_CAN_BUS.vy = EGO_Sensor_Simulated_Data.vy[t]
    EGO_CAN_BUS.yaw = EGO_Sensor_Simulated_Data.yaw[t]
    EGO_CAN_BUS.yawRate = EGO_Sensor_Simulated_Data.yawRate[t]
    EGO_CAN_BUS.detTimeStamp = EGO_Sensor_Simulated_Data.detTimeStamp[t]

    return EGO_CAN_BUS
