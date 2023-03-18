from simulatingCan import *
from preprocess import *

from can import *
from coordinateCluster import *
from statePrediction import *
from model import *
from gating import *
from fusion import *
from trackManger import *
import matplotlib.pyplot as plt

#  =======================================>  SET MODEL PARAMETERS% <=============================================================================================
sigmaQ = 1
SamplingTime = .05
dT = SamplingTime  # %in sec (50 millisec)

# set the process model as constant velocity model
MOTION_MODEL_CV = cvmodel(dT, sigmaQ)
MEAS_MODEL_CV = cvmeasmodelPxPy()  # set the observation model

GammaSq = 9.2103
P_G = 0.99

# radar measurement clustering euclidean distance threshold
epsPosDBSCAN = 5.0
# camera measurement clustering euclidean distance threshold
epsPosNN = 4.0
# radar and camera cluster grouping euclidean distance threshold
epsPosJOIN = 4.0
ANTICLOCKWISE_ROT = 1.0
CLOCKWISE_ROT = -1.0
ROT_CONVENTION = ANTICLOCKWISE_ROT

RadarCALLIBRATIONparam.Intrinsic = setRadarIntrinsicParam(nRadarTypes,
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
                                                          RadarINTRINSICparam)


RadarCALLIBRATIONparam.Extrinsic = setSensorExtrinsicParam(nRadars, RAD_TYPE, ACTIVATE_RAD,
                                                           RAD_X_INSTALL, RAD_Y_INSTALL, RAD_Z_INSTALL,
                                                           RAD_ROLL_INSTALL, RAD_PITCH_INSTALL, RAD_YAW_INSTALL,
                                                           RAD_nMeas, RadarCALLIBRATIONparam.Extrinsic, ROT_CONVENTION)


CameraCALLIBRATIONparam.Intrinsic = setCameraIntrinsicParam(nCameraTypes,
                                                            CAMERA_MAX_RANGE, CAMERA_MAX_AZIMUTH, CAMERA_MAX_ELEVATION,
                                                            CAMERA_LONG_ERR_VAR, CAMERA_LAT_ERR_VAR,
                                                            CAMERA_PD, CAMERA_FA,
                                                            CAMERA_FOV_BOUNDARY_PTS_RANGE, CAMERA_FOV_BOUNDARY_PTS_AZIMUTH,
                                                            CameraINTRINSICparam)


CameraCALLIBRATIONparam.Extrinsic = setSensorExtrinsicParam(nCameras, CAM_TYPE, ACTIVATE_CAM,
                                                            CAM_X_INSTALL, CAM_Y_INSTALL, CAM_Z_INSTALL,
                                                            CAM_ROLL_INSTALL, CAM_PITCH_INSTALL, CAM_YAW_INSTALL,
                                                            CAM_nMeas, CameraCALLIBRATIONparam.Extrinsic, ROT_CONVENTION)


nTimeSample = 401
ExecutionCycleTime_FUS = np.zeros((nTimeSample, 1), dtype=float)
ExecutionCycleTime_CAM = np.zeros((nTimeSample, 1), dtype=float)
ExecutionCycleTime_RAD = np.zeros((nTimeSample, 1), dtype=float)


CAM1_Sensor_Simulated_Data = detectionData[0, 0]
CAM2_Sensor_Simulated_Data = detectionData[1, 0]
CAM3_Sensor_Simulated_Data = detectionData[2, 0]
CAM4_Sensor_Simulated_Data = detectionData[3, 0]
CAM5_Sensor_Simulated_Data = detectionData[4, 0]
CAM6_Sensor_Simulated_Data = detectionData[5, 0]
CAM7_Sensor_Simulated_Data = detectionData[6, 0]
CAM8_Sensor_Simulated_Data = detectionData[7, 0]

RAD1_Sensor_Simulated_Data = detectionData[8, 0]
RAD2_Sensor_Simulated_Data = detectionData[9, 0]
RAD3_Sensor_Simulated_Data = detectionData[10, 0]
RAD4_Sensor_Simulated_Data = detectionData[11, 0]
RAD5_Sensor_Simulated_Data = detectionData[12, 0]
RAD6_Sensor_Simulated_Data = detectionData[13, 0]


for t in range(150):
    print("Time ", t)
    RADAR_CAN_BUS = RAD_SENSOR_INTERFACE(RAD1_Sensor_Simulated_Data,
                                         RAD2_Sensor_Simulated_Data,
                                         RAD3_Sensor_Simulated_Data,
                                         RAD4_Sensor_Simulated_Data,
                                         RAD5_Sensor_Simulated_Data,
                                         RAD6_Sensor_Simulated_Data,
                                         RADAR_CAN_BUS, t, nRadars, nMeas)

    CAMERA_CAN_BUS = CAM_SENSOR_INTERFACE(CAM1_Sensor_Simulated_Data,
                                          CAM2_Sensor_Simulated_Data,
                                          CAM3_Sensor_Simulated_Data,
                                          CAM4_Sensor_Simulated_Data,
                                          CAM5_Sensor_Simulated_Data,
                                          CAM6_Sensor_Simulated_Data,
                                          CAM7_Sensor_Simulated_Data,
                                          CAM8_Sensor_Simulated_Data,
                                          CAMERA_CAN_BUS, t, nCameras, nMeas)

    EGO_CAN_BUS = EGO_SENSOR_INTERFACE(egoData, EGO_CAN_BUS, t)

    RADAR_MEAS_CTS = CTS_SENSOR_FRAME_TO_EGO_FRAME(
        nRadars, nMeas, RadarCALLIBRATIONparam.Extrinsic, RADAR_CAN_BUS, RADAR_MEAS_CTS)

    # SOMETHING WRONG WITH THE CAMERA CTS WHILE MATCHING WITH THE MATLAB ONE, RADAR WAS COMING FINE
    CAMERA_MEAS_CTS = CTS_SENSOR_FRAME_TO_EGO_FRAME(
        nCameras, nMeas, CameraCALLIBRATIONparam.Extrinsic, CAMERA_CAN_BUS, CAMERA_MEAS_CTS)

    RADAR_CLUSTERS, nCountsRadar = findMeanAndVar(
        RADAR_MEAS_CTS, RADAR_CLUSTERS, epsPosDBSCAN=5)
    RADAR_MEAS_CLUSTER = SEGREGATE_CLUSTER(
        RADAR_MEAS_CTS, RADAR_CLUSTERS, nCountsRadar)

    # STATE PREDICTION OF RADAR OBJECTS/TRACKS
    # *******************************************************************************

    TRACK_ESTIMATES_RAD = EGO_COMPENSATION(
        TRACK_ESTIMATES_RAD, EGO_CAN_BUS, dT, None)

    TRACK_ESTIMATES_RAD = LINEAR_PROCESS_MODEL(
        TRACK_ESTIMATES_RAD, MOTION_MODEL_CV)

    # RADAR MEASUREMENT CLUSTER GATING & IDENTIFY UNGATED CLUSTERS
    # *******************************************************************************

    # the function is not returning anything because the references are being modified in place
    GATE_MEASUREMENTS(TRACK_ESTIMATES_RAD, RADAR_MEAS_CLUSTER, MEAS_MODEL_CV,
                      RadarCALLIBRATIONparam, GammaSq, MOTION_MODEL_CV, P_G,
                      ASSOCIATION_MAT_RADAR, ASSIGNMENT_MAT_RADAR, GATED_CLUSTER_INDEX_RAD)

    GATED_MEAS_INDEX_RAD = FIND_GATED_MEASUREMENT_INDEX(
        GATED_MEAS_INDEX_RAD, RADAR_MEAS_CTS, GATED_CLUSTER_INDEX_RAD, RADAR_MEAS_CLUSTER, RADAR_CLUSTERS)

    UNASSOCIATED_CLUSTERS_RAD, cntRadClst = FIND_UNGATED_CLUSTERS(RADAR_MEAS_CTS.ValidCumulativeMeasCount[-1],
                                                                  GATED_MEAS_INDEX_RAD, RADAR_CLUSTERS, UNASSOCIATED_CLUSTERS_RAD, nCountsRadar)

    # RADAR MEASUREMENT , TRACKS  & RADAR SENSOR FUSION
    # *******************************************************************************

    TRACK_ESTIMATES_RAD, FUSION_INFO_RAD = DATA_ASSOCIATION(TRACK_ESTIMATES_RAD, ASSIGNMENT_MAT_RADAR,
                                                            RADAR_MEAS_CLUSTER, MEAS_MODEL_CV, FUSION_INFO_RAD)

    TRACK_ESTIMATES_RAD = HOMOGENEOUS_SENSOR_FUSION_RADARS(
        TRACK_ESTIMATES_RAD, FUSION_INFO_RAD)

    # LOCAL RADAR TRACK MANAGEMENT
    # *******************************************************************************

    TRACK_ESTIMATES_RAD, _ = INIT_NEW_TRACK(
        RADAR_CLUSTERS, UNASSOCIATED_CLUSTERS_RAD, cntRadClst, TRACK_ESTIMATES_RAD, dT)

    TRACK_ESTIMATES_RAD = MAINTAIN_EXISTING_TRACK(TRACK_ESTIMATES_RAD, dT)

    TRACK_ESTIMATES_RAD, _ = DELETE_LOST_TRACK(TRACK_ESTIMATES_RAD, TrackParam)
    print("Number of radar tracks", TRACK_ESTIMATES_RAD.nValidTracks)

    # ===============================================================================
    # STATE PREDICTION OF CAMERA OBJECTS/TRACKS
    # *******************************************************************************

    CAMERA_CLUSTERS, nCountsCamera = findMeanAndVarCAM(
        CAMERA_MEAS_CTS, CAMERA_CLUSTERS, epsPosDBSCAN=5)

    TRACK_ESTIMATES_CAM = EGO_COMPENSATION(
        TRACK_ESTIMATES_CAM, EGO_CAN_BUS, dT, None)

    TRACK_ESTIMATES_CAM = LINEAR_PROCESS_MODEL(
        TRACK_ESTIMATES_CAM, MOTION_MODEL_CV)

   # the function is not returning anything because the references are being modified in place
    GATE_MEASUREMENTS(TRACK_ESTIMATES_CAM, CAMERA_MEAS_CTS, MEAS_MODEL_CV,
                      CameraCALLIBRATIONparam, GammaSq, MOTION_MODEL_CV, P_G,
                      ASSOCIATION_MAT_CAMERA, ASSIGNMENT_MAT_CAMERA, GATED_MEAS_INDEX_CAM)

    UNASSOCIATED_CLUSTERS_CAM, cntCamClst = FIND_UNGATED_CLUSTERS(CAMERA_MEAS_CTS.ValidCumulativeMeasCount[-1],
                                                                  GATED_MEAS_INDEX_CAM, CAMERA_CLUSTERS, UNASSOCIATED_CLUSTERS_CAM, nCountsCamera)

    # ASSOCIATION

    TRACK_ESTIMATES_CAM, FUSION_INFO_CAM = DATA_ASSOCIATION(TRACK_ESTIMATES_CAM, ASSIGNMENT_MAT_CAMERA,
                                                            CAMERA_MEAS_CTS, MEAS_MODEL_CV, FUSION_INFO_CAM)

    TRACK_ESTIMATES_CAM = HOMOGENEOUS_SENSOR_FUSION_RADARS(
        TRACK_ESTIMATES_CAM, FUSION_INFO_CAM)

    # LOCAL TRACK MANAGEMENT CAMERA

    TRACK_ESTIMATES_CAM, _ = INIT_NEW_TRACK(
        CAMERA_CLUSTERS, UNASSOCIATED_CLUSTERS_CAM, cntCamClst, TRACK_ESTIMATES_CAM, dT)

    TRACK_ESTIMATES_CAM = MAINTAIN_EXISTING_TRACK(TRACK_ESTIMATES_CAM, dT)

    TRACK_ESTIMATES_CAM, _ = DELETE_LOST_TRACK(TRACK_ESTIMATES_CAM, TrackParam)
    print("Number of camera tracks ", TRACK_ESTIMATES_CAM.nValidTracks)

    ########################################################## FUSION STUFF  ###############################################################

    # % ------------------------------------------------------ > State Prediction of the object Tracks <----------------------------------------------------------
    TRACK_ESTIMATES_FUS = EGO_COMPENSATION(
        TRACK_ESTIMATES_FUS, EGO_CAN_BUS, dT, None)
    TRACK_ESTIMATES_FUS = LINEAR_PROCESS_MODEL(
        TRACK_ESTIMATES_FUS, MOTION_MODEL_CV)

    # % ----------------------------------------------> Gating local radar and camera tracks with the fused track <-----------------------------------------------
    GATED_TRACK_INFO, UNGATED_TRACK_INFO = GATE_FUSED_TRACK_WITH_LOCAL_TRACKS(
        TRACK_ESTIMATES_FUS, TRACK_ESTIMATES_RAD, TRACK_ESTIMATES_CAM)

    # % ----------------------------------------------------> Sensor Fusion (Heterogeneous,  RADAR + CAMERA) <----------------------------------------------------
    TRACK_ESTIMATES_FUS = TRACK_FUSION_HETEROGENEOUS_SENSORS(
        TRACK_ESTIMATES_FUS, TRACK_ESTIMATES_RAD, TRACK_ESTIMATES_CAM, GATED_TRACK_INFO)

    # % ------------------------------------------------> Set New (FUSED) Tracks from the Clustered Local Tracks <------------------------------------------------
    FUSED_TRACKS, nNewTracks = FORM_NEW_TRACKS_FROM_LOCAL_TRACKS(
        TRACK_ESTIMATES_RAD, TRACK_ESTIMATES_CAM, UNGATED_TRACK_INFO)

    # % ----------------------------------------------------------------> FUSED TRACK MANAGEMENT <----------------------------------------------------------------
    TRACK_ESTIMATES_FUS = SET_NEW_TRACK_INFO(
        TRACK_ESTIMATES_FUS, FUSED_TRACKS, nNewTracks, dT)
    TRACK_ESTIMATES_FUS = MAINTAIN_EXISTING_TRACK(
        TRACK_ESTIMATES_FUS, dT)
    TRACK_ESTIMATES_FUS, _ = DELETE_LOST_TRACK(TRACK_ESTIMATES_FUS, TrackParam)
    print("Number of fused tracks  ", TRACK_ESTIMATES_FUS.nValidTracks)
