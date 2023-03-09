import numpy as np
from dataclasses import dataclass
import copy


def REMOVE_GATING_AMBUIGITY(ASSOCIATION_MAT, nValidTracks, nMeasSnsr):
    # Remove gating ambiguity : It is assumed that each measurement is originated from a single target , i.e no measurements are shared.
    # If a measurement is shared by multiple tracks, the track corresponding to maximum likelihood is kept and the other likelihoods are discarded.
    # Alternate logics include JPDAF, Murtys-k best assignment, Hungarian Assignment, but this one is the simpliest and computationally efficient
    # INPUTS : ASSOCIATION_MAT : Track to measurement association matrix for a single sensor measurements
    #        : nValidTracks : number of valid tracks
    #        : nMeasSnsr : number of valid measurements per sensor
    # OUTPUTS : ASSOCIATION_MAT : Gating Ambiguity removed Track to measurement association matrix
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    INVALID = -99
    for idxMeas in range(nMeasSnsr):
        index = np.argmax(ASSOCIATION_MAT[0:nValidTracks, idxMeas])
        maxLikelihood = ASSOCIATION_MAT[index, idxMeas]
        if(maxLikelihood > INVALID - 1):
            ASSOCIATION_MAT[1:nValidTracks, idxMeas] = INVALID
            ASSOCIATION_MAT[index, idxMeas] = maxLikelihood
    return ASSOCIATION_MAT


def GatingAndLogLikelihood(Z, measmodel, state_pred, P_D, GammaSq):
    # Perform Ellipsoidal Gating and Compute the likelihood of the predicted measurement in logarithmic scale
    # Ellipsoidal Gating is performed individually for (px, py) AND (vx, vy)
    # INPUT :       z : measurement , struct with 2 fields, x: meas vector, R: meas noise covariance
    #        measmodel: a structure specifies the measurement model parameters
    #       state_pred: a structure with two fields:
    #                x: predicted object state mean (state dimension) x 1 vector
    #                P: predicted object state covariance (state dimension) x (state dimension) matrix
    #              P_D: probability of detection
    #  GammaPosSquare : Square of Gamma for position based ellipsoidal gating
    #  GammaVelSquare : Square of Gamma for velocity based ellipsoidal gating
    # OUTPUT : LogLikelihood : log likelihood of the predicted state (valid only if the the state is gated with a measurement)
    #        :       isGated : boolean flag indicating if the measurement is gated
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    INVALID = -99
    x_pred = measmodel.convertToMeasSpace(state_pred.x)
    H = measmodel.H
    S = H.dot(state_pred.P.dot(H.transpose())) + Z.P
    S = (S + S.transpose())/2
    S_inv = np.linalg.inv(S)
    from scipy.spatial import distance
    mDist = distance.mahalanobis(Z.x, x_pred, S_inv)  # mahalanobis distance
    measDim = Z.x.shape[0]  # measurement dimension
    if measDim == 1:
        Ck = 2
    elif measDim == 2:
        Ck = np.pi
    elif measDim == 3:
        Ck = 4*np.pi/3
    elif measDim == 4:
        Ck = np.pi ^ 2/2

    if (mDist <= GammaSq):
        Vk = Ck*np.sqrt(np.linalg.det(GammaSq*S))
        # % - log(numGatedMeas)
        LogLikelihood = np.log(P_D) + np.log(Vk) - 0.5 * \
            np.log(np.sqrt(np.linalg.det(2*np.pi*S))) - 0.5 * \
            mDist   # i have added an additional sqrt for the subtractin term.
        isGated = 1
    else:
        LogLikelihood = INVALID
        isGated = 0
    return LogLikelihood, isGated


def GATE_MEASUREMENTS(Track_Estimates, Measurements, measmodel,
                      SensorParam, GammaSq,  motionmodel, P_G,
                      ASSOCIATION_MAT, ASSIGNMENT_MAT, GATED_MEAS_INDEX):
    # Gating of Sensor measurements , create association matrix for track and sensor measurements
    # separate functions might be needed for radar and camera , because the gating procedure might be different for sensors with different modalities
    # INPUTS : Track_Estimates    : Estimated Track Data Structure
    #        : Measurements       : Coordinate Transformed Radar measurements
    #        : measmodel          : Measurement model
    #        : SensorParam        : sensor intrinsic and extrinsic parameters
    #        : GammaSq            : Gating Threshold
    #        : motionmodel        : Motion Model
    #        : P_G                : probability of gating
    #        : ASSOCIATION_MAT    : (INITIALIZED) Association matrix (nTracks , nTracks + nMeas)
    #        : ASSIGNMENT_MAT     : (INITIALIZED) Structure holding the ASSOCIATION_MAT for each radar sensors
    #        : GATED_MEAS_INDEX   : (INITIALIZED) boolean flag array indicating gated measurement indexes
    # OUTPUT : ASSIGNMENT_MAT     : (UPDATED) Structure holding the ASSOCIATION_MAT for each radar sensors
    #          GATED_MEAS_INDEX   : (UPDATED) boolean flag array indicating gated measurement indexes
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nValidTracks = Track_Estimates.nValidTracks
    # Total number of clusters
    nMeas = Measurements.ValidCumulativeMeasCount[-1]
    GATED_MEAS_INDEX[:] = 0
    for idx in range(len(ASSIGNMENT_MAT)):
        ASSIGNMENT_MAT[idx].AssociationMat[:] = 0.0
        ASSIGNMENT_MAT[idx].nMeas = 0

    # do not perform gating if no objects or no measurements are present
    if (nValidTracks == 0 or nMeas == 0):
        return
    INVALID = -99

    @dataclass
    class state:
        x: np.array = np.zeros((4,), dtype=float)
        P: np.array = np.zeros((4, 4), dtype=float)

    statePred = state()
    z = state()
    covIndex = [0, 1, 3, 4]
    for snsrIdx in range(len(SensorParam.Extrinsic)):
        # extract the number of measurements returned by the sensor # this is because sometimes a single sensor can return multiple clusters
        nMeasSnsr = Measurements.ValidMeasCount[0, snsrIdx]
        if snsrIdx == 0:  # % compute the measurement index offset in the measurement matrix
            startIndexOffet = 0
        else:
            startIndexOffet = Measurements.ValidCumulativeMeasCount[snsrIdx-1]

        P_D = SensorParam.Intrinsic[SensorParam.Extrinsic[snsrIdx,
                                                          0].SensorType - 1][0].ProbOfDetection  # we are doing sensor type -1 because sensor type is 1 and 2
        # and the indexes in python start from zero, so we will get an out of bound index error if we are not doing so.
        ASSOCIATION_MAT[0:nValidTracks, 0:(
            int(nValidTracks) + int(nMeasSnsr))] = INVALID
        for objIdx in range(nValidTracks):

            # here we will assume a constant velocity model
            statePred.x = [Track_Estimates.TrackParam[objIdx].StateEstimate.px,
                           Track_Estimates.TrackParam[objIdx].StateEstimate.vx,
                           Track_Estimates.TrackParam[objIdx].StateEstimate.py,
                           Track_Estimates.TrackParam[objIdx].StateEstimate.vy]
            # THIS IS NOT THE CORRECT COVARIANCE REWRITE THIS
            #statePred.P = Track_Estimates.TrackParam[objIdx].StateEstimate.ErrCOV[:4, :4]

            P = Track_Estimates.TrackParam[objIdx].StateEstimate.ErrCOV  # 6x6
            row1 = P[[covIndex[0]], covIndex]
            row2 = P[[covIndex[1]], covIndex]
            row3 = P[[covIndex[2]], covIndex]
            row4 = P[[covIndex[3]], covIndex]
            statePred.P = np.array([row1, row2, row3, row4])

            # nMeasSnsr is usually one , but it can be greater than one when we have a radar detecting two objects.
            nGatedMeas = 0
            # cost for missdetection
            ASSOCIATION_MAT[objIdx, int(
                objIdx) + int(nMeasSnsr)] = np.log(1-P_D*P_G)
            for idxMeas in range(nMeasSnsr):
                measIndex = startIndexOffet + idxMeas
                z.x = Measurements.MeasArray[:, measIndex]
                z.P = Measurements.MeasCovariance[:, :, measIndex]
                LogLikelihood, isGated = GatingAndLogLikelihood(
                    z, measmodel, statePred, P_D, GammaSq)
                ASSOCIATION_MAT[objIdx, idxMeas] = LogLikelihood
                if(isGated):
                    GATED_MEAS_INDEX[0, measIndex] = isGated
                    nGatedMeas = nGatedMeas + 1

        ASSOCIATION_MAT = REMOVE_GATING_AMBUIGITY(
            ASSOCIATION_MAT, nValidTracks, nMeasSnsr)  # % Resolve the Gate
        ASSIGNMENT_MAT[snsrIdx].AssociationMat = copy.deepcopy(ASSOCIATION_MAT)
        ASSIGNMENT_MAT[snsrIdx].nMeas = nMeasSnsr


def FIND_GATED_MEASUREMENT_INDEX(GATED_MEAS_INDEX, SENSOR_MEASUREMENTS, GATED_CLUSTER_INDEX, CLUSTER_MEASUREMENTS, CLUSTERS):
    # extract the measurement indexes from the gated measurement clusters
    # INPUTS : GATED_CLUSTER_INDEX  : initialized gated measurememnt index array
    #          SENSOR_MEASUREMENTS  : sensor measurements (for details refer to 'SensorFusion_Script3_LOAD_DATA_STRUCTURE_PARAMETERS')
    #          GATED_CLUSTER_INDEX  : index of the gated clusters
    #          CLUSTER_MEASUREMENTS : Measurement clusters from each sensors
    #          CLUSTERS             : measurement clusters from all sensors
    # OUTPUTS: GATED_MEAS_INDEX     : gated measurememnt index array updated
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    GATED_MEAS_INDEX[:] = 0
    nSnsrClusters = CLUSTER_MEASUREMENTS.ValidCumulativeMeasCount[-1]
    nSnsrMeas = SENSOR_MEASUREMENTS.ValidCumulativeMeasCount[-1]

    # GATED_CLSTR_INDEX_LIST = #find(GATED_CLUSTER_INDEX(1,1:nSnsrClusters) ~= 0);   #list of gated radar cluster index
    GATED_CLSTR_INDEX_LIST = np.where(
        GATED_CLUSTER_INDEX[0, :nSnsrClusters] != 0)[0]
    # GATED_CLSTR_LIST = #CLUSTER_MEASUREMENTS.ClusterRef(1,GATED_CLSTR_INDEX_LIST); #list of gated radar cluster ID
    # %list of gated radar cluster ID
    GATED_CLSTR_LIST = CLUSTER_MEASUREMENTS.ClusterRef[0,
                                                       GATED_CLSTR_INDEX_LIST]
    # GATED_CLSTR_LIST = #unique(GATED_CLSTR_LIST)
    GATED_CLSTR_LIST = np.unique(GATED_CLSTR_LIST)
    if(GATED_CLSTR_LIST.shape[0] > 0):
        for idx in range(len(GATED_CLSTR_LIST)):
            GATED_MEAS_INDEX_temp = np.where(
                CLUSTERS.ClustIDAssig[0, 0:nSnsrMeas] == GATED_CLSTR_LIST[idx])[0]
            GATED_MEAS_INDEX[0, GATED_MEAS_INDEX_temp] = 1
    return GATED_MEAS_INDEX


def FIND_UNGATED_CLUSTERS(nSnsrMeas, GATED_MEAS_INDEX, CLUSTERS_MEAS, UNASSOCIATED_CLUSTERS, nCounts):
    # Find the list of unassociated radar and camera cluster centers
    # INPUTS : nSnsrMeas             : number of sensor measurements
    #          GATED_MEAS_INDEX      : gated measurememnt index array
    #          CLUSTERS_MEAS         : Measurement clusters from each sensors
    #          UNASSOCIATED_CLUSTERS : Unassociated clusters initialized data structure
    # OUTPUTS: UNASSOCIATED_CLUSTERS : Unassociated clusters updated data structure
    #          cntMeasClst           : number of ungated easurement clusters
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    UNASSOCIATED_CLUSTERS[:] = 0
    cntMeasClst = 0
    if (nSnsrMeas != 0):
        UNGATED_MEAS_INDEX_LIST = np.where(GATED_MEAS_INDEX[0, :nSnsrMeas] == 0)[
            0].tolist()  # list of ungated sensor index
        # number of ungated radar meas
        nUngatedMeas = len(UNGATED_MEAS_INDEX_LIST)
        isMeasVisited = [False for i in range(nSnsrMeas)]

        for idx in range(nUngatedMeas):
            ungatedMeasIdx = UNGATED_MEAS_INDEX_LIST[idx]
            if(not isMeasVisited[idx]):
                # Cluster ID
                clusterID = CLUSTERS_MEAS.ClustIDAssig[0, ungatedMeasIdx]
                UNASSOCIATED_CLUSTERS[0, cntMeasClst] = clusterID
                MeasList = np.where(CLUSTERS_MEAS.ClustIDAssig[0, :nCounts] == clusterID)[
                    0]        # find the measurement index with the same radar cluster ID
                for val in MeasList:
                    isMeasVisited[val] = True
                cntMeasClst = cntMeasClst + 1
    return UNASSOCIATED_CLUSTERS, cntMeasClst


def GATE_FUSED_TRACK_WITH_LOCAL_TRACKS(TRACK_ESTIMATES_FUS, TRACK_ESTIMATES_RAD, TRACK_ESTIMATES_CAM):
    # % Gating of fused tracks with the local tracks
    # % INPUTS : TRACK_ESTIMATES_FUS   : Fused Track predictions
    # %          TRACK_ESTIMATES_RAD   : Local Track estimates from Radar sensor
    # %          TRACK_ESTIMATES_CAM   : Local Track estimates from Camera sensor
    # % OUTPUTS: GATED_TRACK_INFO      : Unassociated clusters updated data structure
    # %          UNGATED_TRACK_INFO    : number of ungated easurement clusters
    # % --------------------------------------------------------------------------------------------------------------------------------------------------
    # % initialize the Gated info
    maxNumFusedTracks = 100
    maxNumLocalTracks = 100

    @dataclass
    class CGATED_TRACK_INFO:
        nGatedRadarTracks: int = 0
        nGatedCameraTracks: int = 0
        RadarTracks: np.array = np.zeros((1, maxNumLocalTracks), dtype=int)
        CameraTracks: np.array = np.zeros((1, maxNumLocalTracks))
    GATED_TRACK_INFO = [copy.deepcopy(CGATED_TRACK_INFO())
                        for i in range(maxNumFusedTracks)]

    @dataclass
    class CUNGATED_TRACK_INFO:
        UngatedRadarTracks = np.ones((1, maxNumLocalTracks), dtype=int)
        UngatedCameraTracks = np.ones((1, maxNumLocalTracks), dtype=int)
    UNGATED_TRACK_INFO = CUNGATED_TRACK_INFO()

    # if the local tracks are available and the fused tracks are also available then execute this function
    if(((TRACK_ESTIMATES_RAD.nValidTracks == 0) and (TRACK_ESTIMATES_CAM.nValidTracks == 0)) or (TRACK_ESTIMATES_FUS.nValidTracks == 0)):
        return GATED_TRACK_INFO, UNGATED_TRACK_INFO

    # % init structures
    GammaSqPos = 16  # %GammaSqVel = 10
    posCovIdx = [0, 3]
    velCovIdx = [1, 4]

    @dataclass
    class CFusState:
        x: np.array = np.zeros((2, 1))
        P: np.array = np.zeros((2, 2))

    xFusPos = copy.deepcopy(CFusState())
    xFusVel = copy.deepcopy(CFusState())
    xLocalPos = copy.deepcopy(CFusState())
    xLocalVel = copy.deepcopy(CFusState())

    # % create the association matrix for radar local tracks
    for i in range(TRACK_ESTIMATES_FUS.nValidTracks):
        nGatedTracksRad = 0
        nGatedTracksCam = 0
        nGatedTracksRad = 0
        nGatedTracksCam = 0
        xFusPos.x[0, 0] = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.px
        xFusPos.x[1, 0] = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.py
        # THIS IS WRONG NEED TO BE CHANGED
        xFusPos.P = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.ErrCOV[:2, :2]
        xFusVel.x[0, 0] = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.vx
        xFusVel.x[1, 0] = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.vy
        # THIS IS WRONG NEED TO BE CHANGED
        xFusVel.P = TRACK_ESTIMATES_FUS.TrackParam[i].StateEstimate.ErrCOV[:2, :2]

        for j in range(TRACK_ESTIMATES_RAD.nValidTracks):  # % for each of the radar tracks
            xLocalPos.x[0, 0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.px
            xLocalPos.x[1, 0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.py
            # THIS IS WRONG NEED TO BE CHANGED
            xLocalPos.P = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
            xLocalVel.x[0, 0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.vx
            xLocalVel.x[1, 0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.vy
            # THIS IS WRONG NEED TO BE CHANGED
            xLocalVel.P = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
            dist = np.sqrt((xFusPos.x[0, 0] - xLocalPos.x[0, 0])
                           ** 2 + (xFusPos.x[1, 0] - xLocalPos.x[1, 0])**2)
            if(dist <= np.sqrt(GammaSqPos)):  # % if Gated set the gated track info
                GATED_TRACK_INFO[i].RadarTracks[0, nGatedTracksRad] = j
                UNGATED_TRACK_INFO.UngatedRadarTracks[0, j] = False
                nGatedTracksRad = nGatedTracksRad + 1

        for j in range(TRACK_ESTIMATES_CAM.nValidTracks):  # % for each of the camera tracks
            xLocalPos.x[0, 0] = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.px
            xLocalPos.x[1, 0] = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.py
            xLocalPos.P = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
            xLocalVel.x[0, 0] = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.vx
            xLocalVel.x[1, 0] = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.vy
            xLocalVel.P = TRACK_ESTIMATES_CAM.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
            dist = np.sqrt((xFusPos.x[0, 0] - xLocalPos.x[0, 0])
                           ** 2 + (xFusPos.x[1, 0] - xLocalPos.x[1, 0])**2)
            if(dist <= np.sqrt(GammaSqPos)):
                GATED_TRACK_INFO[i].CameraTracks[0, nGatedTracksCam] = j
                UNGATED_TRACK_INFO.UngatedCameraTracks[0, j] = False
                nGatedTracksCam = nGatedTracksCam + 1

        # % Update the Gated meas count info
        GATED_TRACK_INFO[i].nGatedRadarTracks = nGatedTracksRad
        GATED_TRACK_INFO[i].nGatedCameraTracks = nGatedTracksCam

    return GATED_TRACK_INFO, UNGATED_TRACK_INFO
