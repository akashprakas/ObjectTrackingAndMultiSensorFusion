import numpy as np
from dataclasses import dataclass
import copy


def momentMatching(weights, X):
    # Approximate a Gaussian mixture density as a single Gaussian using moment matching
    # INPUT: weights: normalised weight of Gaussian components in logarithm domain (number of Gaussians) x 1 vector
    #              X: structure array of size (number of Gaussian components x 1), each structure has two fields
    #           mean: means of Gaussian components (variable dimension) x 1 vector
    #              P: variances of Gaussian components (variable dimension) x (variable dimension) matrix
    # OUTPUT:  state: a structure with two fields:
    #              x: approximated mean (variable dimension) x 1 vector
    #              P: approximated covariance (variable dimension) x (variable dimension) matrix
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nComponents = len(weights)
    if nComponents == 1:
        x = X[0].x
        P = X[0].P
        return x, P
    mixtureMean = 0
    Paverage = 0
    meanSpread = 0
    # % convert normalized weights from log scale to linear scale (exp(w))
    w = np.exp(weights)
    # compute the weighted average of the means (in the gaussian mixture)
    for idx in range(nComponents):
        mixtureMean = mixtureMean + w[idx] * X[idx].x

    # % compute weighted average covariance and spread of the mean
    for idx in range(nComponents):
        # % compute weighted average covariance
        Paverage = Paverage + w[idx] * X[idx].P
        std = (mixtureMean - X[idx].x).reshape(4, 1)
        meanSpread = meanSpread + \
            w[idx]*(std.dot(std.transpose()))   # spread of the mean

    x = mixtureMean
    P = Paverage + meanSpread
    return x, P


def normalizeLogWeights(LogWeights):
    # Normalize the weights in log scale
    # INPUT  :    LogWeights: log weights, e.g., log likelihoods
    # OUTPUT :    LogWeights: log of the normalized weights
    #        : sumLogWeights: log of the sum of the non-normalized weights
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    if len(LogWeights) == 1:
        sumLogWeights = LogWeights
        LogWeights = LogWeights - sumLogWeights
        return LogWeights, sumLogWeights
    # we need to sort in descending order and arguments are also needed so sorting the negative of actual thing to get in descending order
    Index = np.argsort(-LogWeights)
    logWeights_aux = - np.sort(-LogWeights)
    sumLogWeights = max(logWeights_aux) + np.log(1 +
                                                 sum(np.exp(LogWeights[Index[1:]] - max(logWeights_aux))))
    LogWeights = LogWeights - sumLogWeights  # % normalize
    return LogWeights, sumLogWeights


def InovationCovAndKalmanGain(statePred, measmodel, R):
    # Computes the innovation covariance and Kalman Gain
    # INPUT: state: a structure with two fields:
    #            x: object state mean (dim x 1 vector)
    #            P: object state covariance ( dim x dim matrix )
    #        z: measurement
    #        R: measurement noise covariance
    #        measmodel: a structure specifies the measurement model parameters
    # OUTPUT: S: Innovation Covariance
    #         K: Kalman Gain
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    H = measmodel.H  # Measurement model Jacobian
    S = H.dot(statePred.P.dot(H.transpose())) + R
    S = (S+S.transpose())/2
    K = (statePred.P.dot(H.transpose())).dot(np.linalg.inv(S))
    return S, K


def update(state_pred, z, K, measmodel):
    # performs linear/nonlinear (Extended) Kalman update step
    # INPUT: z: measurement (measurement dimension) x 1 vector
    #        state_pred: a structure with two fields:
    #                x: predicted object state mean (state dimension) x 1 vector
    #                P: predicted object state covariance (state dimension) x (state dimension) matrix
    #        K: Kalman Gain
    #        measmodel: a structure specifies the measurement model parameters
    # OUTPUT:state_upd: a structure with two fields:
    #                   x: updated object state mean (state dimension) x 1 vector
    #                   P: updated object state covariance (state dimension) x (state dimension) matrix
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    @dataclass
    class state:
        x: np.array = np.zeros((4, 1))
        P: np.array = np.zeros((4, 4))
    state_upd = state()
    Hx = measmodel.H  # measurement model jacobian
    state_upd.x = state_pred.x + \
        K.dot(z - measmodel.convertToMeasSpace(state_pred.x))  # State update
    # Covariance update
    state_upd.P = (np.eye(state_pred.x.shape[0]) - K.dot(Hx)).dot(state_pred.P)
    return state_upd


def AssociationHypothesis(TrackPred, ASSIGNMENT_MAT, MEAS_CTS, measmodel, snsrIdx, idxObj):
    # Compute Association log probabilities for a Track (single sensor measurements)
    # INPUTS : TrackPred: a structure with two fields:
    #                  x: predicted object state mean (state dimension) x 1 vector
    #                  P: predicted object state covariance (state dimension) x (state dimension) matrix
    #          ASSOCIATION_MAT: structure with the following fields (single sensor measurements):
    #          AssociationMat : association matrix of log likelihoods
    #                   nMeas : number of sensor measurements
    #                MEAS_CTS : coordinate transformed measurements all sensors
    #                snsrIdx  : sensor index
    #                idxObj   : Track index
    # OUTPUTS : Beta : Normalized Log probabilities
    #           BetaSum : sum of log probabilities before normalization (product of probabilities)
    #           MixtureComponents : Gaussian mixture components after association
    #           nHypothesis : number of Hypothesis
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nHypothesis = 0  # % count total number of hypothesis

    @dataclass
    class CComponents:
        x: np.array = np.zeros((4, 1))
        P: np.array = np.zeros((4, 4))

    Components = CComponents()
    MixtureComponents = [CComponents() for i in range(200)]
    Beta = np.zeros((1, 200))
    INVALID_WT = -99
    nMeasSnsr = ASSIGNMENT_MAT.nMeas
    if snsrIdx == 0:  # % compute the measurement index offset in the measurement matrix
        startIndexOffet = 0
    else:
        startIndexOffet = MEAS_CTS.ValidCumulativeMeasCount[snsrIdx-1]

    # nHypothesis = nHypothesis + 1
    MixtureComponents[nHypothesis] = TrackPred
    Beta[0, nHypothesis] = ASSIGNMENT_MAT.AssociationMat[idxObj,
                                                         int(idxObj) + int(nMeasSnsr)]  # miss detection weight
    TrackUpd = TrackPred
    for idxMeas in range(nMeasSnsr):

        if(ASSIGNMENT_MAT.AssociationMat[idxObj, idxMeas] != INVALID_WT):
            idx = startIndexOffet + idxMeas
            z = MEAS_CTS.MeasArray[:, idx]
            R = MEAS_CTS.MeasCovariance[:, :, idx]
            _, K = InovationCovAndKalmanGain(
                TrackPred, measmodel, R)
            TrackUpd = update(
                TrackPred, z, K, measmodel)  # kalman filter update
            nHypothesis = nHypothesis + 1
            MixtureComponents[nHypothesis] = TrackUpd
            Beta[0, nHypothesis] = ASSIGNMENT_MAT.AssociationMat[
                idxObj, idxMeas]  # %detection weight substracting one so that the indexing will work properly.
    [Beta, BetaSum] = normalizeLogWeights(Beta[0, 0: nHypothesis+1])
    return Beta, BetaSum, MixtureComponents, nHypothesis, TrackUpd


def DATA_ASSOCIATION(TRACK_DATA_in, ASSIGNMENT_MAT, MEAS_CTS, measmodel, FUSION_INFO):
    # Data Association using Radar sensor measurements
    # INPUTS : TRACK_DATA_in  : Track Data structure
    #        : ASSIGNMENT_MAT : Track to Measurement association matrix
    #        : MEAS_CTS       : Coordinate Transformed sensor measurements
    #        : measmodel      : measurement model
    #        : FUSION_INFO    : Initialized Structure to hold Data association results
    # OUTPUT : TRACK_DATA     : Track Data
    #        : FUSION_INFO    : Data association results
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    TRACK_DATA = TRACK_DATA_in

    @dataclass
    class state:
        x: np.array = np.zeros((4,), dtype=float)
        P: np.array = np.zeros((4, 4), dtype=float)
    TrackPred = state()
    for row in range(FUSION_INFO.shape[0]):
        for col in range(FUSION_INFO.shape[1]):
            FUSION_INFO[row, col].Beta[:] = 0.0
            FUSION_INFO[row, col].BetaSum = 0.0
            FUSION_INFO[row, col].nHypothesis = 0.0

    # NEED TO FILL MORE
    covIndex = [0, 1, 3, 4]
    for idxObj in range(TRACK_DATA.nValidTracks):
        TrackPred.x = np.array([TRACK_DATA.TrackParam[idxObj].StateEstimate.px,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.vx,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.py,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.vy])
        # CHECK AND SEE IF THE COVS ARE COMING CORRECTLY
        P_ = TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV
        row1 = P_[[covIndex[0]], covIndex]
        row2 = P_[[covIndex[1]], covIndex]
        row3 = P_[[covIndex[2]], covIndex]
        row4 = P_[[covIndex[3]], covIndex]

        TrackPred.P = np.array([row1, row2, row3, row4])

        for snsrIdx in range(len(ASSIGNMENT_MAT)):
            Beta, BetaSum, MixtureComponents, nHypothesis, TrackUpdated = AssociationHypothesis(TrackPred, ASSIGNMENT_MAT[snsrIdx],
                                                                                                MEAS_CTS,  measmodel, snsrIdx, idxObj)
            FUSION_INFO[idxObj, snsrIdx].Beta = Beta
            FUSION_INFO[idxObj, snsrIdx].BetaSum = BetaSum
            FUSION_INFO[idxObj, snsrIdx].MixtureComponents = MixtureComponents
            # adding plus one because the index start at zero and the prediction is considered as one hyopthesis even if there is no update
            FUSION_INFO[idxObj, snsrIdx].nHypothesis = nHypothesis+1
            FUSION_INFO[idxObj, snsrIdx].x = TrackUpdated.x
            FUSION_INFO[idxObj, snsrIdx].P = TrackUpdated.P

    return TRACK_DATA, FUSION_INFO


def HOMOGENEOUS_SENSOR_FUSION_RADARS(TRACK_DATA_in, FUSION_INFO_RAD):
    # Sensor Fusion with multiple radars
    # INPUTS : TRACK_DATA_in   : Track Data structure
    #        : FUSION_INFO_RAD : Data association results from each Radars
    # OUTPUT : TRACK_DATA      : Data Fusion results (all Radar sensors)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    TRACK_DATA = TRACK_DATA_in
    if(TRACK_DATA.nValidTracks == 0):

        return TRACK_DATA

    covIndex = [0, 1, 3, 4]
    # %for state err covariance(corresponding to px, vx, py, vy)
    nRadars = FUSION_INFO_RAD.shape[1]  # % number of cameras

    for idxObj in range(TRACK_DATA_in.nValidTracks):
        # initialize the hypothesis and other things initially
        validIndex = np.zeros((1, nRadars), dtype=int)
        nHypothesis = np.zeros((1, nRadars), dtype=int)
        Beta = np.zeros((1, nRadars))
        # initialize some parameters to zeros
        TRACK_DATA.TrackParam[idxObj].Status.Predicted = False
        TRACK_DATA.TrackParam[idxObj].Status.Gated = False
        TRACK_DATA.TrackParam[idxObj].SensorSource.RadarCatch = False
        TRACK_DATA.TrackParam[idxObj].SensorSource.RadarSource[:] = False
        count = 0
        for idxSnsr in range(nRadars):
            # initialize some parameters to zeros
            Beta[0, idxSnsr] = FUSION_INFO_RAD[idxObj, idxSnsr].BetaSum
            nHypothesis[0, idxSnsr] = FUSION_INFO_RAD[idxObj,
                                                      idxSnsr].nHypothesis
        if (int(np.sum(nHypothesis)) == nRadars):  # % only prediction
            MixtureComponents = FUSION_INFO_RAD[idxObj, 0:nRadars]
            Beta, _ = normalizeLogWeights(Beta)
            Xfus, Pfus = momentMatching(
                Beta, MixtureComponents)
            TRACK_DATA.TrackParam[idxObj].StateEstimate.px = Xfus[0]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.vx = Xfus[1]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.py = Xfus[2]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.vy = Xfus[3]
            # NEED TO CHANGE THIS PART OF THE CODE!!!!
            # TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV[:4,
            #                                                    :4] = Pfus
            P_ = TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV
            P_[[covIndex[0]], covIndex] = Pfus[0]
            P_[[covIndex[1]], covIndex] = Pfus[1]
            P_[[covIndex[2]], covIndex] = Pfus[2]
            P_[[covIndex[3]], covIndex] = Pfus[3]

            TRACK_DATA.TrackParam[idxObj].Status.Predicted = True
            TRACK_DATA.TrackParam[idxObj].Status.Gated = False
            TRACK_DATA.TrackParam[idxObj].SensorSource.RadarCatch = False
            TRACK_DATA.TrackParam[idxObj].SensorSource.RadarSource[:] = False

        elif (int(np.sum(nHypothesis)) > nRadars):  # % merge estimates from different sensors
            for idxSnsr in range(nRadars):
                # % consider only those state estimate which was updated by at least one measurement
                if(FUSION_INFO_RAD[idxObj, idxSnsr].nHypothesis > 1):
                    validIndex[0, count] = idxSnsr
                    count = count + 1
                    TRACK_DATA.TrackParam[idxObj].SensorSource.RadarSource[idxSnsr] = 1.0
            MixtureComponents = FUSION_INFO_RAD[idxObj, validIndex[0, 0:count]]
            Beta, _ = normalizeLogWeights(
                Beta[0, validIndex[0, 0:count]])
            Xfus, Pfus = momentMatching(
                Beta, MixtureComponents)
            TRACK_DATA.TrackParam[idxObj].StateEstimate.px = Xfus[0]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.vx = Xfus[1]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.py = Xfus[2]
            TRACK_DATA.TrackParam[idxObj].StateEstimate.vy = Xfus[3]
            # NEED TO CHANGE THIS PART OF THE CODE!!!!
            P_ = TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV
            P_[[covIndex[0]], covIndex] = Pfus[0]
            P_[[covIndex[1]], covIndex] = Pfus[1]
            P_[[covIndex[2]], covIndex] = Pfus[2]
            P_[[covIndex[3]], covIndex] = Pfus[3]

            # TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV[:4,
            #    :4] = Pfus
            TRACK_DATA.TrackParam[idxObj].Status.Predicted = False
            TRACK_DATA.TrackParam[idxObj].Status.Gated = True
            TRACK_DATA.TrackParam[idxObj].SensorSource.RadarCatch = True

    return TRACK_DATA


def CovarianceIntersection(Xr, W):
    # % Fusion of Gaussian Distributions by Covarience Intersection method
    # % INPUT : W  : normalised weight of Gaussian components (row vector : 1 x N )
    # %       : Xr : structure array of size (1 x N), each structure has two fields
    # %          x : mean
    # %          P : covariance
    # % OUTPUT : X : mean vector by covariance intersection
    # %        : P : covariance intersection matrix
    # % --------------------------------------------------------------------------------------------------------------

    if(len(W) == 1):
        X = Xr[0, 0].x
        P = Xr[0, 0].P
        return X, P

    dim = Xr.shape[1]
    Pavginv = 0
    temp = 0
    I = np.eye(dim)
    # NOTE EVERYTHING BELOW IS WRONG AND NEED TO BE RECALCULATED
    for i in range(len(W)):
        Pavginv = Pavginv + W[0, i] * ((Xr[0, i].P).dot(I))
        temp = temp + W[0, i] * ((Xr[0, i].P).dot(Xr[0, i].x))

    P = Pavginv
    X = P*temp


def TRACK_FUSION_HETEROGENEOUS_SENSORS(TRACK_ESTIMATES_FUS_in, TRACK_ESTIMATES_RAD, TRACK_ESTIMATES_CAM, GATED_TRACK_INFO):
    # % Sensor Fusion with multiple cameras
    # % INPUTS : TRACK_ESTIMATES_FUS_in : Fused Track Data structure
    # %        : TRACK_ESTIMATES_RAD    : Local Track estimates from Radar sensors
    # %        : TRACK_ESTIMATES_CAM    : Local Track estimates from Camera sensors
    # %        : GATED_TRACK_INFO       : Local Track to Fused Track gating info
    # % OUTPUT : TRACK_ESTIMATES_FUS    : Data Fusion results (Radar Tracks + Camera Tracks)
    # % --------------------------------------------------------------------------------------------------------------------------------------------------
    # % execute this function only if there are valid tracks

    TRACK_ESTIMATES_FUS = copy.deepcopy(TRACK_ESTIMATES_FUS_in)
    if(TRACK_ESTIMATES_FUS.nValidTracks == 0):
        return TRACK_ESTIMATES_FUS

    # % initializations
    StateCovIndex = [0, 1, 3, 4]
    nLocalTracksCam = 100
    nLocalTracksRad = 100
    nLocalTracks = nLocalTracksCam + nLocalTracksRad
    dim = 4

    @dataclass
    class CXTracks:

        x = np.zeros((dim, 1))
        P = np.zeros((dim, dim))

    XTracks = [CXTracks() for _ in nLocalTracks]
    weights = np.zeros((1, dim))
    CameraSource = 0
    RadarSource = 0
   # @% Association of the local tracks with the fused track
    # % for each of the predicted fused tracks
    for idx in range(TRACK_ESTIMATES_FUS.nValidTracks):
        count = 0
        # % number of gated radar local tracks
        nRadGatedTracks = GATED_TRACK_INFO[idx].nGatedRadarTracks
        # % number of gated camera local tracks
        nCamGatedTracks = GATED_TRACK_INFO[idx].nGatedCameraTracks
        # % total number of gated radar + camera local tracks
        nGatedTracks = nRadGatedTracks + nCamGatedTracks
        GatedTrack = False
        PredictedTrack = True
        RadarCatch = False
        CameraCatch = False
        CameraSource = False
        RadarSource = False
        for i in range(nRadGatedTracks):

            j = GATED_TRACK_INFO(idx).RadarTracks(1, i)
            XTracks[count].x[0,
                             0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.px
            XTracks[count].x[1,
                             0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.vx
            XTracks[count].x[2,
                             0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.py
            XTracks[count].x[3,
                             0] = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.vy
            XTracks[count].P = TRACK_ESTIMATES_RAD.TrackParam[j].StateEstimate.ErrCOV[:4, :4]
            # % updat varios flags here
            RadarCatch = (
                RadarCatch or TRACK_ESTIMATES_RAD.TrackParam[j].SensorSource.RadarCatch)
            RadarSource = (
                RadarSource or TRACK_ESTIMATES_RAD.TrackParam[j].SensorSource.RadarSource)
            GatedTrack = (
                GatedTrack or TRACK_ESTIMATES_RAD.TrackParam[j].Status.Gated)
            PredictedTrack = (
                PredictedTrack and TRACK_ESTIMATES_RAD.TrackParam[j].Status.Predicted)
            count = count + 1

        for i in range(nCamGatedTracks):

            j = GATED_TRACK_INFO(idx).CameraTracks(1, i)
            XTracks[count].x[0, 1] = TRACK_ESTIMATES_CAM.TrackParam(
                1, j).StateEstimate.px
            XTracks[count].x[1, 1] = TRACK_ESTIMATES_CAM.TrackParam(
                1, j).StateEstimate.vx
            XTracks[count].x[2, 1] = TRACK_ESTIMATES_CAM.TrackParam(
                1, j).StateEstimate.py
            XTracks[count].x[3, 1] = TRACK_ESTIMATES_CAM.TrackParam(
                1, j).StateEstimate.vy
            XTracks[count].P = TRACK_ESTIMATES_CAM.TrackParam(
                1, j).StateEstimate.ErrCOV[:4, :4]
            # % update various flags here
            CameraCatch = (
                CameraCatch or TRACK_ESTIMATES_CAM.TrackParam[j].SensorSource.CameraCatch)
            CameraSource = (
                CameraSource or TRACK_ESTIMATES_CAM.TrackParam[j].SensorSource.CameraSource)
            GatedTrack = (
                GatedTrack or TRACK_ESTIMATES_CAM.TrackParam[j].Status.Gated)
            PredictedTrack = (
                PredictedTrack and TRACK_ESTIMATES_CAM.TrackParam[j].Status.Predicted)
            count = count + 1

        if(nGatedTracks == 0):  # % only prediction , no fusion
            # % update various counters here
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarCatch = False
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.CameraCatch = False
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarAndCameraCatch = False
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarSource[:] = False
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.CameraSource[:] = False
            TRACK_ESTIMATES_FUS.TrackParam[idx].Status.Predicted = True
            TRACK_ESTIMATES_FUS.TrackParam[idx].Status.Gated = False

        elif(nGatedTracks > 0):  # % else track fusion
            # % assign equal weights to all local sensor estimates
            weights[0, 1:count] = 1/nGatedTracks
            Xfus, Pfus = CovarianceIntersection(
                XTracks[0, 0:count], weights[0, 1:count])
            TRACK_ESTIMATES_FUS.TrackParam[idx].StateEstimate.px = Xfus(1, 1)
            TRACK_ESTIMATES_FUS.TrackParam[idx].StateEstimate.vx = Xfus(2, 1)
            TRACK_ESTIMATES_FUS.TrackParam[idx].StateEstimate.py = Xfus(3, 1)
            TRACK_ESTIMATES_FUS.TrackParam[idx].StateEstimate.vy = Xfus(4, 1)
            TRACK_ESTIMATES_FUS.TrackParam[idx].StateEstimate.ErrCOV[:4, :4] = Pfus
            RadarCameraCatch = (RadarCatch and CameraCatch)
            # % update various counters here
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.CameraCatch = CameraCatch
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarCatch = RadarCatch
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarCameraCatch = RadarCameraCatch
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.CameraSource = CameraSource
            TRACK_ESTIMATES_FUS.TrackParam[idx].SensorSource.RadarSource = RadarSource
            TRACK_ESTIMATES_FUS.TrackParam[idx].Status.Predicted = PredictedTrack
            TRACK_ESTIMATES_FUS.TrackParam[idx].Status.Gated = GatedTrack

    return TRACK_ESTIMATES_FUS
