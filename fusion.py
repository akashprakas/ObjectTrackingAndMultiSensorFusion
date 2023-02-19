import numpy as np
from dataclasses import dataclass


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
    logWeights_aux = - np.sort(LogWeights)
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
    return Beta, BetaSum, MixtureComponents, nHypothesis, TrackPred


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
    StateCovIndex = [0, 1, 3, 4]
    for idxObj in range(TRACK_DATA.nValidTracks):
        TrackPred.x = np.array([TRACK_DATA.TrackParam[idxObj].StateEstimate.px,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.vx,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.py,
                                TRACK_DATA.TrackParam[idxObj].StateEstimate.vy])
        # NEED TO REMOVE THIS COVARIANCE AND REPLACE WITH THE CORRECT ONE
        TrackPred.P = TRACK_DATA.TrackParam[idxObj].StateEstimate.ErrCOV[:4, :4]

        for snsrIdx in range(len(ASSIGNMENT_MAT)):
            Beta, BetaSum, MixtureComponents, nHypothesis, TrackPred = AssociationHypothesis(TrackPred, ASSIGNMENT_MAT[snsrIdx],
                                                                                             MEAS_CTS,  measmodel, snsrIdx, idxObj)
            FUSION_INFO[idxObj, snsrIdx].Beta = Beta
            FUSION_INFO[idxObj, snsrIdx].BetaSum = BetaSum
            FUSION_INFO[idxObj, snsrIdx].MixtureComponents = MixtureComponents
            FUSION_INFO[idxObj, snsrIdx].nHypothesis = nHypothesis
            FUSION_INFO[idxObj, snsrIdx].x = TrackPred.x
            FUSION_INFO[idxObj, snsrIdx].P = TrackPred.P

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
    # NEED TO FILL MORE
