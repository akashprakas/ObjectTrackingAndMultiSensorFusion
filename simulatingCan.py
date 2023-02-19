import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from numpy import sin, cos, arctan2
import copy

nMeas = 200  # maximum number of sensor measurements possible (per sensor)
nRadars = 6  # number of radars installed around the subject vehicle
nCameras = 8  # number of cameras installed around the subject vehicle
# number of types of radars (as of now two types of radars : LRR and MRR)
nRadarTypes = 2
nCameraTypes = 2  # number of types of cameras (NFOV, WFOV)
nTracks = 100   # maximum number of estimated object tracks possible
# measurement dimension for radar and camera measurements (px, vx, py, vy) % only range measurements (px, py) are used
dimVector = 2
# number of attributes to maintain other info (meas index in CAN and sensor ID)
dimInfo = 2
# maximum number of measurements (all radar sensors)
maxNumMeasRadar = nRadars*nMeas
# maximum number of measurements (all camera sensors)
maxNumMeasCamera = nCameras*nMeas
maxNumMeasUpperLimit = max(maxNumMeasRadar, maxNumMeasCamera)
# number of Points sampled on Sensor Field of View boundary (on one side of the FOV symmetry)
nFOVpoints = 20
nTrackHistoryPoints = 100  # number of trajectory History points
nPointsSengment = 20  # number of trajectory History points to fit
nSegments = nTrackHistoryPoints/nPointsSengment


@dataclass
class CegoCanBus:
    detTimeStamp: float
    px: float
    py: float
    vx: float
    vy: float
    yaw: float
    yawRate: float


EGO_CAN_BUS = CegoCanBus(0., 0., 0., 0., 0., 0., 0.)
noise = np.zeros((2, 2), dtype=float)

#Sensor = namedtuple('Sensor','sensorID time px py vx vy classID SNR ErrCov')
# Set Radar Sensor Interface (An Array of structure whose purpose is to hold the Radar measurements at current time t)


@dataclass
class CRadarCanBus:

    measID: int = 0
    sensorID: int = 0
    detTimeStamp: float = 0.0
    px: float = 0.0
    py: float = 0.0
    measNoise: np.array = noise
    snr: float = 0.0


RADAR_CAN_BUS = np.empty((nRadars, nMeas), dtype=object)
for i in range(nRadars):
    for j in range(nMeas):
        RADAR_CAN_BUS[i, j] = CRadarCanBus()

# Set Camera Sensor Interface (An Array of structure whose purpose is to hold the Radar measurements at current time t)


@dataclass
class CcameraCanBus:

    measID: int = 0
    sensorID: int = 0
    detTimeStamp: float = 0.0
    px: float = 0.0
    py: float = 0.0
    measNoise: np.array = noise
    objClassID: int = 0


CAMERA_CAN_BUS = np.empty((nCameras, nMeas), dtype=object)
for i in range(nCameras):
    for j in range(nMeas):
        CAMERA_CAN_BUS[i, j] = CcameraCanBus()


zero_matrix = np.zeros((2, 2), dtype=float)
zero_vector = np.zeros((2, 1), dtype=float)


# Sensor Extrinsic Parameters Structure (Applicable for both Radar and Camera)
@dataclass
class CSensorEXTRINSICparam:
    SensorID: int = 0
    SensorType: int = 0
    isActive: bool = False
    RotMat2D: np.array((2, 2)) = zero_matrix
    TranslationVec: np.array((2, 1)) = zero_vector
    MountX: float = 0.0
    MountY: float = 0.0
    MountZ: float = 0.0
    MountYaw: float = 0.0
    MountPitch: float = 0.0
    MountRoll: float = 0.0
    nMeas: float = 0.0


RadarEXTRINSICparam = np.empty((nRadars, 1), dtype=object)
CameraEXTRINSICparam = np.empty((nCameras, 1), dtype=object)

for i in range(nRadars):
    RadarEXTRINSICparam[i] = CSensorEXTRINSICparam()

for i in range(nCameras):
    CameraEXTRINSICparam[i] = CSensorEXTRINSICparam()

fovVec = np.zeros((1, nFOVpoints), dtype=float)


@dataclass
class CRadarINTRINSICparam:
    RadarType: int = 0
    MaxRange: float = 0.0
    MaxAzimuth: float = 0.0
    MaxElevation: int = 0
    MaxRangeRate: float = 0.0
    RangeResolution: float = 0.0
    AzimuthResolution: float = 0.0
    ElevationResolution: float = 0.0
    RangeRateResolution: float = 0.0
    RangeErrVariance: float = 0.0
    AzimuthErrVariance: float = 0.0
    ElevationErrVariance: int = 0
    RangeRateErrVariance: float = 0.0
    ProbOfDetection: float = 0.0
    FalseAlarmRate: float = 0.0
    FOVRangePoints: np.array((1, nFOVpoints), dtype=float) = fovVec
    FOVAzimuthPts: np.array((1, nFOVpoints), dtype=float) = fovVec


RadarINTRINSICparam = np.empty((nRadarTypes, 1), dtype=object)
# each radartype (we have 2) will have its own of intrinsic paramaters
for i in range(nRadarTypes):
    RadarINTRINSICparam[i] = CRadarINTRINSICparam()


@dataclass
class CCameraINTRINSICparam:
    CameraType: int = 0
    RectificationMatrix: float = 0.0
    ProjectionMatrix: float = 0.0
    MaxRange: float = 0.0
    MaxAzimuth: float = 0.0
    MaxElevation: int = 0
    LongitudinalErrVariance: float = 0.0
    LateralErrVariance: float = 0.0
    ProbOfDetection: float = 0.0
    FalseAlarmRate: float = 0.0
    FOVRangePoints: np.array((1, nFOVpoints), dtype=float) = fovVec
    FOVAzimuthPts: np.array((1, nFOVpoints), dtype=float) = fovVec


# here also we have two types of cameras so we will need two intinsic param list
CameraINTRINSICparam = np.empty((nCameraTypes, 1), dtype=object)
for i in range(nCameraTypes):
    CameraINTRINSICparam[i] = CCameraINTRINSICparam()


class CCALLIBRATIONparam:
    def __init__(self, Intrinsic, Extrinsic):
        self.Intrinsic = Intrinsic
        self.Extrinsic = Extrinsic


RadarCALLIBRATIONparam = CCALLIBRATIONparam(
    RadarINTRINSICparam, RadarEXTRINSICparam)
CameraCALLIBRATIONparam = CCALLIBRATIONparam(
    CameraINTRINSICparam, CameraEXTRINSICparam)


# STATE PARAMS
# -----------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class StateParam:
    px: float = 0.0  # longitudinal position estimate of the track
    py: float = 0.0  # lateral position estimate of the track
    vx: float = 0.0  # longitudinal velocity estimate of the track
    vy: float = 0.0  # lateral velocity estimate of the track
    ax: float = 0.0  # longitudinal acceleration estimate of the track
    ay: float = 0.0  # lateral acceleration estimate of the track
    ErrCOV: np.array((6, 6), dtype=float) = np.zeros((6, 6), dtype=float)


@dataclass
class StateParamAccuracy:
    px3Sigma: float = 0.0  # 3 sigma error for the longitudinal position estimate of the track
    py3Sigma: float = 0.0  # 3 sigma error for the lateral position estimate of the track
    vx3Sigma: float = 0.0  # 3 sigma error for the longitudinal velocity estimate of the track
    vy3Sigma: float = 0.0  # 3 sigma error for the lateral velocity estimate of the track
    # 3 sigma error for the longitudinal acceleration estimate of the track
    ax3Sigma: float = 0.0
    ay3Sigma: float = 0.0  # 3 sigma error for the lateral acceleration estimate of the track


@dataclass
class StateParamCAM:
    Classification: int = 0  # classification of the object/bounding box
    ClassificationScore: float = 0.0  # classification score of the object
    BoundingBoxDim: np.array((3, 1), dtype=float) = np.zeros(
        (3, 1), dtype=float)  # in the order of length width and height
    BoundingBoxPoint: np.array((3, 1), dtype=float) = np.zeros(
        (3, 1), dtype=float)  # in the order of X, Y, Z


# Sensor Source of the Track (which sensors contributed to the track estimation)
@dataclass
class SensorSource:
    # radar sensor sources that contributed to state update
    RadarSource: np.array((nRadars, 1)) = np.array(
        [[0.], [0], [0], [0], [0], [0]])
    CameraSource: np.array((nCameras, 1)) = np.array([[0.], [0], [0], [0], [0], [
        0], [0], [0.]])  # camera sensor sources that contributed to state update
    RadarCatch: bool = False  # is the state updated from radar measurements
    CameraCatch: bool = False  # is the state updated from the camera measurements
    # is the state updated from both radar and camera measurements
    RadarAndCameraCatch: bool = False


# Track status (is Occluded, is Out Of FOV, is in prediction mode, is lost , is stationary, is an obstacle etc)
@dataclass
class isTrack:
    # Indicates if a track is not confirmed for a newly appeared object
    New: bool = False
    # Indicates if the track is confirmed for an object, and is updated with a measurement
    Existing: bool = False
    # Indicates if the track it is not of interest anymore and needs to be deleted
    Lost: bool = False
    # Indicates if the track is predicted (not gated or updated with any measurements)
    Predicted: bool = False
    Gated: bool = False       # Indicate  if the track is gated with any measurement
    OutOfFov: bool = False    # Indicates if the track is out of FoV of all active sensors
    Occluded: bool = False    # Indicates if the track is occluded
    Stationary: bool = False  # Indicates if the object track is stationary
    SlowMoving: bool = False  # Indicates if the object track is moving slowly
    Moving: bool = False      # Indicates if the object track is moving
    # Indicates if the track is a part of static environment (guardrails, barriers, parked-vehicles, etc)
    Obstacle: bool = False
    # Indicates if the track is capable of moving independently (pedstrians, 2-wheelers, 4-wheelers etc#
    Object: bool = False


SamplingTime = 0.0500


@dataclass
class Track:
    TrackedTime: float = -SamplingTime    # Total tracked time in sec
    RegionConfidence: float = 0           # Region Confidence of the Track
    Quality: int = 0                       # Track estimate quality
    # number of times the track got predicted continuously
    PredictedCounter: int = 0
    # number of times the track got gated within a certain time interval
    GatedCounter: int = 0


class CTrackParam:
    def __init__(self, id, StateEstimate, StateEstimateAccuracy, BoundingBoxInfo, SensorSource, Status, Quality):
        self.id = id
        self.StateEstimate = StateEstimate
        self.StateEstimateAccuracy = StateEstimateAccuracy
        self.BoundingBoxInfo = BoundingBoxInfo
        self.SensorSource = SensorSource
        self.Status = Status
        self.Quality = Quality


TrackParam = CTrackParam(0, StateParam(), StateParamAccuracy(),
                         StateParamCAM(), SensorSource(), isTrack(), Track())


class CTRACK_ESTIMATES:
    def __init__(self, nTracks, nValidTracks=0):
        self.nValidTracks = nValidTracks  # number of valid tracks in time t
        # Track estimated/computed parameters
        self.TrackParam = [copy.deepcopy(CTrackParam(0, StateParam(), StateParamAccuracy(),
                                                     StateParamCAM(), SensorSource(), isTrack(), Track())) for i in range(nTracks)]
        # list of all possible track ids
        self.TrackIDList = [i for i in range(nTracks)]
        # are there track ids assigned to any track
        self.IsTrackIDused = [0 for i in range(nTracks)]
        # index to the first available track id (track ids not currently used by existing tracks)
        self.FirstAvailableIDindex = 0
        # index to the last available track id (track ids not currently used by existing tracks)
        self.LastAvailableIDindex = nTracks


TRACK_ESTIMATES = CTRACK_ESTIMATES(nTracks, 0)


@dataclass
class DTRACK_HISTORY:
    id: int = 0
    TrackEstRefIdx: int = 0
    BufferStartIndex: int = 0  # number of elements held in the buffer
    WriteIndex: int = 1  # buffer array index where the latest element has to be copied
    SegmentLength: float = 0  # trajectory segment length
    Length: float = 0  # trajectory length
    isInitialized: bool = False  # is the trajectory buffer initialized
    HistoryBufferPx: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    HistoryBufferPy: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    HistoryBufferVx: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    HistoryBufferVy: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    HistoryBufferAx: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    HistoryBufferAy: np.array = np.zeros((1, nTrackHistoryPoints), dtype=float)
    # 4 3rd degree polynomial coefficients : curve length, c0, c1, c2, c3
    SegmentedPolynomialCoefficients: np.array = np.zeros(
        (5, int(nSegments)), dtype=float)
    # curve length, initial offset , heading, curvature and curvature rate
    SegmentedClothoidCoefficients: np.array = np.zeros(
        (5, int(nSegments)), dtype=float)
    # merged clothoid curve coefficients : init offset,
    ClothoidCoefficients: np.array = np.zeros((1, 4+int(nSegments)))
    SegmentAvailableToFit: int = 0  # indicates how many segments are available to fit


class CTRACK_HISTORY:
    def __init__(self, dataClass):
        self.TRACK_HISTORY = [dataClass for i in range(int(nTracks))]


TRACK_HISTORY = CTRACK_HISTORY(DTRACK_HISTORY)


class CTRAJECTORY_HISTORY:
    def __init__(self, nConfirmedTracks, TRACK_HISTORY):
        self.nConfirmedTracks = 0
        self.TRACK_HISTORY = TRACK_HISTORY


TRAJECTORY_HISTORY = CTRAJECTORY_HISTORY(0, TRACK_HISTORY)

# for Track to Track fusion
# track estimates from the radar sensor
TRACK_ESTIMATES_RAD = CTRACK_ESTIMATES(nTracks, 0)
# track estimates from the camera sensor
TRACK_ESTIMATES_CAM = CTRACK_ESTIMATES(nTracks, 0)
# track fusion estimates from the radar and camera estimated tracks
TRACK_ESTIMATES_FUS = CTRACK_ESTIMATES(nTracks, 0)


# # Sensor Measurement Noise
# ### Structure of Sensor Measurement matrix :

#  MeasArray : Sensor measurement matrix holding coordinate transformed measurements for all sensor measuurements :

#        : size : (meas_dim, max_num_of_meas_all_sensors)

#  MeasCovariance : Sensor measurement matrix holding coordinate transformed meas noise covariance for all sensor measuurements :

#        : size : (meas_dim, meas_dim, max_num_of_meas_all_sensors)

# MeasRef : Sensor measurement reference matrix holding measurement index and sensor index of the measurement CAN bus

#        : size : (2, max_num_of_meas_all_sensors)

# ValidMeasCount : maximum number of valid measurements returned by each of the sensors

#        : size : (1, number of sensors)

# ValidCumulativeMeasCount : vector of cumulative count of 'ValidMeasCount'

#         : size : (1, number of sensors)


@dataclass
class RADAR_MEAS_CTS:
    MeasArray: np.array = np.zeros(
        (dimVector, int(nRadars)*nMeas), dtype=float)
    MeasCovariance: np.array = np.zeros(
        (dimVector, dimVector, int(nRadars)*nMeas), dtype=float)
    MeasRef: np.array = np.zeros((dimInfo, int(nRadars)*nMeas), dtype=int)
    ValidMeasCount: np.array = np.zeros((1, nRadars), dtype=int)
    ValidCumulativeMeasCount: np.array = np.zeros((1, nRadars), dtype=int)


@dataclass
class CAMERA_MEAS_CTS:
    MeasArray: np.array = np.zeros(
        (dimVector, int(nCameras)*nMeas), dtype=float)
    MeasCovariance: np.array = np.zeros(
        (dimVector, dimVector, int(nCameras)*nMeas), dtype=float)
    MeasRef: np.array = np.zeros((dimInfo, int(nCameras)*nMeas), dtype=int)
    ValidMeasCount: np.array = np.zeros((1, nCameras), dtype=int)
    ValidCumulativeMeasCount: np.array = np.zeros((1, nCameras), dtype=int)


@dataclass
class RADAR_CLUSTERS:
    nClusters: int = 0  # number of Radar clusters
    # Radar cluster sizes (number of measurements forming a cluster)
    ClusterSizes: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)
    ClusterCenters: np.array = np.zeros(
        (dimVector, maxNumMeasRadar), dtype=float)  # Radar cluster centers
    # Radar cluster center errror covariance
    ClusterCovariance: np.array = np.zeros(
        (dimVector, dimVector, maxNumMeasRadar), dtype=float)
    ClusterIDs: np.array = np.zeros(
        (1, maxNumMeasRadar), dtype=int)  # Radar cluster Id
    # Radar measurement to Radar cluster ID assignment vector
    ClustIDAssig: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)


@dataclass
class CAMERA_CLUSTERS:
    nClusters: int = 0  # number of Camera clusters
    # Camera cluster sizes (number of measurements forming a cluster)
    ClusterSizes: np.array = np.zeros((1, maxNumMeasCamera), dtype=int)
    ClusterCenters: np.array = np.zeros(
        (dimVector, maxNumMeasCamera), dtype=float)  # Camera cluster centers
    # Camera cluster center errror covariance
    ClusterCovariance: np.array = np.zeros(
        (dimVector, dimVector, maxNumMeasRadar), dtype=float)
    ClusterIDs: np.array = np.zeros(
        (1, maxNumMeasCamera), dtype=int)  # Camera cluster Id
    # Camera measurement to Radar cluster ID assignment vector
    ClustIDAssig: np.array = np.zeros((1, maxNumMeasCamera), dtype=int)


# Clustering Output Structure ( Fused Cluster : Radar + Camera )
# each cluster can have either camera only measurements, radar only measurements, both camera and radar measurements

@dataclass
class MERGED_CLUSTERS:
    nClusters: int = 0  # Total number of measurement clusters
    ClusterCenters: np.array = np.zeros(
        (dimVector, maxNumMeasUpperLimit), dtype=float)  # Merged cluster centers
    # Merged cluster center errror covariance
    ClusterCovariance: np.array = np.zeros(
        (dimVector, dimVector, maxNumMeasUpperLimit), dtype=float)
    ClusterIDs: np.array = np.zeros(
        (1, maxNumMeasUpperLimit), dtype=int)  # Merged cluster Id
    # Radar Cluster ID to Merged cluster ID assignment vector
    LookUpToRadClstrID: np.array = np.zeros(
        (1, maxNumMeasUpperLimit), dtype=int)
    # Camera Cluster ID to Merged cluster ID assignment vector.
    LookUpToCamClstrID: np.array = np.zeros(
        (1, maxNumMeasUpperLimit), dtype=int)


# Unassociated Clusters
# Unassociated Radar clusters to track
UNASSOCIATED_CLUSTERS_RAD = np.zeros((1, maxNumMeasRadar))
# Unassociated Camera clusters to track
UNASSOCIATED_CLUSTERS_CAM = np.zeros((1, maxNumMeasCamera))


# Cluster Manage Structure (RADAR)
@dataclass
class RAD_CLST:
    measTypeCore: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)
    clusterMember: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)
    clusterMemberFlag: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)
    measurementVisited: np.array = np.zeros((1, maxNumMeasRadar), dtype=int)


# Cluster Manage Structure (CAMERA)
@dataclass
class CAM_CLST:
    sizeClust: int = 0
    ClusterID: int = 0
    clusterMember: np.array = np.zeros((1, maxNumMeasCamera), dtype=int)
    measurementVisited: np.array = np.zeros((1, maxNumMeasCamera), dtype=int)

# Cluster Manage Structure (MERGE)


@dataclass
class FUS_CLST:
    RadarClstAdded: np.array = np.zeros((1, maxNumMeasUpperLimit))
    RadarClstrMemberList: np.array = np.zeros((1, maxNumMeasUpperLimit))


# MANAGE CLUSTER
class CMANAGE_CLUSTERS:
    def __init__(self):
        self.RAD = RAD_CLST
        self.CAM = CAM_CLST
        self.FUSE = FUS_CLST


MANAGE_CLUSTERS = CMANAGE_CLUSTERS()


# Boolean Flag array of gated measurement indexes
GATED_MEAS_INDEX_RAD = np.zeros((1, maxNumMeasRadar), dtype=int)
GATED_MEAS_INDEX_CAM = np.zeros((1, maxNumMeasCamera), dtype=int)
GATED_CLUSTER_INDEX_RAD = np.zeros((1, maxNumMeasRadar), dtype=int)
GATED_CLUSTER_INDEX_CAM = np.zeros((1, maxNumMeasCamera), dtype=int)

# Track To Measurement Association Matrix
INV = -99.0
ASSOCIATION_MAT_RADAR = INV * \
    np.zeros((nTracks, (nTracks + maxNumMeasRadar)), dtype=float)
ASSOCIATION_MAT_CAMERA = INV * \
    np.zeros((nTracks, (nTracks + maxNumMeasCamera)), dtype=float)
ASSOCIATION_MAT_UNASSOCIATED_CLUSTER_IDs = np.zeros(
    (1, maxNumMeasUpperLimit), dtype=int)


# Measurement To Track Assignment Matrix
@dataclass
class ASSIGNMENT_MAT:
    AssociationMat: np.array = np.zeros(
        (nTracks, (nTracks + nMeas)), dtype=float)
    nMeas: int = 0


ASSIGNMENT_MAT_RADAR = [ASSIGNMENT_MAT() for i in range(nRadars)]
ASSIGNMENT_MAT_CAMERA = [ASSIGNMENT_MAT() for i in range(nCameras)]


@dataclass
class Components:
    x: np.array = np.zeros((4, 1), dtype=float)
    P: np.array = np.zeros((4, 4), dtype=float)


MixtureComponents = [Components for i in range(nMeas)]


@dataclass
class FusedStates:
    x: np.array = np.zeros((4, 1), dtype=float)
    P: np.array = np.zeros((4, 4), dtype=float)


class CFUSION_INFO:
    def __init__(self):
        self.Beta = np.zeros((1, nMeas))
        self.BetaSum = 0.0
        self.MixtureComponents = MixtureComponents
        self.nHypothesis = 0
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.zeros((4, 4), dtype=float)


FUSION_INFO_RAD = np.empty((nTracks, nRadars), dtype=object)
FUSION_INFO_CAM = np.empty((nTracks, nCameras), dtype=object)


for i in range(nTracks):
    for j in range(nRadars):
        FUSION_INFO_RAD[i, j] = CFUSION_INFO()

for i in range(nTracks):
    for j in range(nCameras):
        FUSION_INFO_CAM[i, j] = CFUSION_INFO()


# SENSOR CONFIG PARAMETERS
# -----------------------------------------------------------------------------------------------------------------------------------------
# RADAR Long Range Radar (LRR) Intrinsic Parameters (Field Of View and Maximum Range)
RADAR_TYPE_LRR = 1
RADAR_LRR_Max_Range = 200
RADAR_LRR_Max_Azimuth = 15
RADAR_LRR_Max_Elevation = -1  # %not currently set
RADAR_LRR_Max_RangeRate = -1  # %not currently set
RADAR_LRR_Max_RangeRes = -1  # %not currently set
RADAR_LRR_Max_AzimuthRes = -1  # %not currently set
RADAR_LRR_Max_ElevationRes = -1  # %not currently set
RADAR_LRR_Max_RangeRateRes = -1  # %not currently set
RADAR_LRR_Max_RangeErr = -1  # %not currently set
RADAR_LRR_Max_AzimuthErr = -1  # %not currently set
RADAR_LRR_Max_ElevationErr = -1  # %not currently set
RADAR_LRR_Max_RangeRateErr = -1  # %not currently set
RADAR_LRR_Max_ProbOfDetection = 0.99
RADAR_LRR_Max_FalseAlarmRate = 2/45000
RADAR_LRR_Max_FovRangePts = np.zeros((1, 20))  # %not currently set
RADAR_LRR_Max_FovAzimuthPts = np.zeros((1, 20))  # %not currently set


# RADAR Long Range Radar (MRR) Intrinsic Parameters (Field Of View and Maximum Range)
RADAR_TYPE_MRR = 2
RADAR_MRR_Max_Range = 100
RADAR_MRR_Max_Azimuth = 50
RADAR_MRR_Max_Elevation = -1  # %not currently set
RADAR_MRR_Max_RangeRate = -1  # %not currently set
RADAR_MRR_Max_RangeRes = -1  # %not currently set
RADAR_MRR_Max_AzimuthRes = -1  # %not currently set
RADAR_MRR_Max_ElevationRes = -1  # %not currently set
RADAR_MRR_Max_RangeRateRes = -1  # %not currently set
RADAR_MRR_Max_RangeErr = -1  # %not currently set
RADAR_MRR_Max_AzimuthErr = -1  # %not currently set
RADAR_MRR_Max_ElevationErr = -1  # %not currently set
RADAR_MRR_Max_RangeRateErr = -1  # %not currently set
RADAR_MRR_Max_ProbOfDetection = 0.99
RADAR_MRR_Max_FalseAlarmRate = 2/45000
RADAR_MRR_Max_FovRangePts = np.zeros((1, 20))  # %not currently set
RADAR_MRR_Max_FovAzimuthPts = np.zeros((1, 20))  # %not currently set


# Sensor Intrinsic Parameters (Field Of View and Maximum Range)
RADAR_MAX_RANGE = [RADAR_LRR_Max_Range, RADAR_MRR_Max_Range]
RADAR_MAX_AZIMUTH = [RADAR_LRR_Max_Azimuth, RADAR_MRR_Max_Azimuth]
RADAR_MAX_ELEVATION = [RADAR_LRR_Max_Elevation, RADAR_MRR_Max_Elevation]
RADAR_MAX_RANGE_RATE = [RADAR_LRR_Max_RangeRate, RADAR_MRR_Max_RangeRate]
RADAR_RANGE_RES = [RADAR_LRR_Max_RangeRes, RADAR_MRR_Max_RangeRes]
RADAR_AZIMUTH_RES = [RADAR_LRR_Max_AzimuthRes, RADAR_MRR_Max_AzimuthRes]
RADAR_ELEVATION_RES = [RADAR_LRR_Max_ElevationRes, RADAR_MRR_Max_ElevationRes]
RADAR_RANGE_RATE_RES = [RADAR_LRR_Max_RangeRateRes, RADAR_MRR_Max_RangeRateRes]
RADAR_RANGE_ERR_VAR = [RADAR_LRR_Max_RangeErr, RADAR_MRR_Max_RangeErr]
RADAR_AZIMUTH_ERR_VAR = [RADAR_LRR_Max_AzimuthErr, RADAR_MRR_Max_AzimuthErr]
RADAR_ELEVATION_ERR_VAR = [
    RADAR_LRR_Max_ElevationErr, RADAR_MRR_Max_ElevationErr]
RADAR_RANGE_RATE_ERR_VAR = [
    RADAR_LRR_Max_RangeRateErr, RADAR_MRR_Max_RangeRateErr]
RADAR_PD = [RADAR_LRR_Max_ProbOfDetection, RADAR_MRR_Max_ProbOfDetection]
RADAR_FA = [RADAR_LRR_Max_FalseAlarmRate, RADAR_MRR_Max_FalseAlarmRate]
RADAR_FOV_BOUNDARY_PTS_RANGE = [
    RADAR_LRR_Max_FovRangePts, RADAR_MRR_Max_FovRangePts]
RADAR_FOV_BOUNDARY_PTS_AZIMUTH = [
    RADAR_LRR_Max_FovAzimuthPts, RADAR_MRR_Max_FovAzimuthPts]


# CAMARA Long Range (Narrow FOV : NFOV) Intrinsic Parameters (Field Of View and Maximum Range)
CAMERA_TYPE_NARROW = 1
CAM_NARROW_Max_Range = 150
CAM_NARROW_Max_Azimuth = 30
CAM_NARROW_Max_Elevation = -1  # % not used
CAM_NARROW_LongErrVar = -1  # % not used
CAM_NARROW_LatErrVar = -1  # % not used
CAM_NARROW_ProbOfDetection = 0.99  # % not used
CAM_NARROW_FalseAlarmRate = 2/45000  # % not used
CAM_NARROW_FOVRangePoints = np.zeros((1, 20))  # % not used
CAM_NARROW_FOVAzimuthPts = np.zeros((1, 20))  # % not used


# CAMARA Short Range (Wide FOV : WFOV) Intrinsic Parameters (Field Of View and Maximum Range)
CAMERA_TYPE_WIDE = 2
CAM_WIDE_Max_Range = 70
CAM_WIDE_Max_Azimuth = 60
CAM_WIDE_Max_Elevation = -1  # % not used
CAM_WIDE_LongErrVar = -1  # % not used
CAM_WIDE_LatErrVar = -1  # % not used
CAM_WIDE_ProbOfDetection = 0.99  # % not used
CAM_WIDE_FalseAlarmRate = 2/45000  # % not used
CAM_WIDE_FOVRangePoints = np.zeros((1, 20), dtype=float)  # ; % not used
CAM_WIDE_FOVAzimuthPts = np.zeros((1, 20), dtype=float)  # ; % not used


# CAMARA Intrinsic Parameters (Field Of View and Maximum Range)
CAMERA_MAX_RANGE = [CAM_NARROW_Max_Range, CAM_WIDE_Max_Range]
CAMERA_MAX_AZIMUTH = [CAM_NARROW_Max_Azimuth, CAM_WIDE_Max_Azimuth]
CAMERA_MAX_ELEVATION = [CAM_NARROW_Max_Elevation, CAM_WIDE_Max_Elevation]
CAMERA_LONG_ERR_VAR = [CAM_NARROW_LongErrVar, CAM_WIDE_LongErrVar]
CAMERA_LAT_ERR_VAR = [CAM_NARROW_LatErrVar, CAM_WIDE_LatErrVar]
CAMERA_PD = [CAM_NARROW_ProbOfDetection, CAM_WIDE_ProbOfDetection]
CAMERA_FA = [CAM_NARROW_FalseAlarmRate, CAM_WIDE_FalseAlarmRate]
CAMERA_FOV_BOUNDARY_PTS_RANGE = [
    CAM_NARROW_FOVRangePoints, CAM_WIDE_FOVRangePoints]
CAMERA_FOV_BOUNDARY_PTS_AZIMUTH = [
    CAM_NARROW_FOVAzimuthPts, CAM_WIDE_FOVAzimuthPts]


# Sensor Extrinsic Parameter (X, Y, Z , Yaw mounting parameters) w.r.t ego vehicle center (in Clock Wise Direction)
RAD_X_INSTALL = [3.7, 3.7,  -1, -1, -1, 3.7]
RAD_Y_INSTALL = [0,  -0.9, -0.9,  0,  0.9,  0.9]
RAD_Z_INSTALL = [0,   0,     0,     0,  0,    0]  # % not set
RAD_YAW_INSTALL = [0,  -45,   -135, -180, 135,  45]
RAD_ROLL_INSTALL = [0,   0,     0,     0,  0,    0]  # % not set
RAD_PITCH_INSTALL = [0,   0,     0,     0,  0,    0]  # % not set
RAD_nMeas = [200, 200, 200, 200, 200, 200]
RAD_TYPE = [RADAR_TYPE_LRR, RADAR_TYPE_MRR, RADAR_TYPE_MRR,
            RADAR_TYPE_MRR, RADAR_TYPE_MRR, RADAR_TYPE_MRR]


# Camera installs in clockwise direction (Check the order once a data is generated)
CAM_X_INSTALL = [1.9, 0,    2.8,     0,  1.38,  2.8,    0, 1.44]
CAM_Y_INSTALL = [0,   0,   -0.9,  -0.9,  -0.9,  0.9,  0.9,   0.9]
CAM_Z_INSTALL = [0,   0,      0,     0,     0,    0,    0,     0]  # % not set
CAM_YAW_INSTALL = [0,  -180,  -60,  -120,   -90,   60,  120,    90]
CAM_ROLL_INSTALL = [0,  0,    0,    0,    0,    0,    0,   0]  # % not set
CAM_PITCH_INSTALL = [0, 0,    0,    0,    0,    0,    0,   0]  # % not set
CAM_nMeas = [200, 200, 200, 200, 200, 200, 200, 200]
CAM_TYPE = [CAMERA_TYPE_NARROW, CAMERA_TYPE_NARROW, CAMERA_TYPE_WIDE, CAMERA_TYPE_WIDE,
            CAMERA_TYPE_WIDE, CAMERA_TYPE_WIDE, CAMERA_TYPE_WIDE, CAMERA_TYPE_WIDE]


# Sensor Layout(SL) module parameters (This module is one time computation which gets actived using a "trigger")
# Sensor Activation Flags for 6 Radars and 8 Cameras (1:Sensor Active, 0:Sensor Inactive)
ACTIVATE_RAD = [1, 1, 1, 1, 1, 1]
ACTIVATE_CAM = [1, 1, 1, 1, 1, 1, 1, 1]
