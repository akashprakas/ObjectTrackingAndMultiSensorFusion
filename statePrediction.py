import numpy as np
from numpy import sin, cos


def EGO_COMPENSATION(TRACK_STATE_ESTIMATES_in, EGO_SENSOR_DATA, dT, model=None):
    # Perform ego compensation of the state from t-1 before prediction. Due to the yaw rate of the ego vehicle, the ego
    # vehicle would have turned by an angle yaw, to represent the prediction in the current ego vehicle frame , we compensate
    # the track position from t-1 by an angle deltaYAW
    # INPUTS: TRACK_STATE_ESTIMATES : Structure of the track state estimates from the previous cycle
    #         nValidObj             : number of valid objects
    #         EGO_SENSOR_DATA       : ego vehicle data
    #         dT                    : sample time
    #         model                 : indicates if ego compensation needs to be applied to acceleration also
    # OUTPUT: TRACK_STATE_ESTIMATES : ego compensated track state estimates
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    TRACK_STATE_ESTIMATES = TRACK_STATE_ESTIMATES_in
    if (model == None):  # do not perform ego compensation
        return TRACK_STATE_ESTIMATES
    dYaw = EGO_SENSOR_DATA.yawRate * dT
    DEG2RAD = np.pi/180
    dYaw = dYaw*DEG2RAD
    yawRateCompensationMat = np.array([[cos(-dYaw), -sin(-dYaw)],
                                       [sin(-dYaw),  cos(-dYaw)]])

    for idx in range(TRACK_STATE_ESTIMATES.nValidTracks):
        # px,py, vx,vy, ax,ay ego compensation
        if(model == 'cv'):
            x = [TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.px,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vx,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.py,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vy]
            # x = STATE_PREDICTOR.EgoCompensationCV(yawRateCompensationMat, x) #HAVE TO IMPLEMENT IT LATER NOW NOT USING

        elif(model == 'ca'):

            x = [TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.px,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vx,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ax,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.py,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vy,
                 TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ay]
            # x = STATE_PREDICTOR.EgoCompensationCA(yawRateCompensationMat, x)   #HAVE TO IMPLEMENT IT LATER NOW NOT USING

        if(model == 'cv'):

            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.px = x[1, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vx = x[2, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.py = x[3, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vy = x[4, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ax = 0.0
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ay = 0.0
        elif (model == 'ca'):
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.px = x[1, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vx = x[2, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ax = x[3, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.py = x[4, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.vy = x[5, 1]
            TRACK_STATE_ESTIMATES.TrackParam[idx].StateEstimate.ay = x[6, 1]

    return TRACK_STATE_ESTIMATES


def LINEAR_PROCESS_MODEL(TRACK_STATE_ESTIMATES_in, motionmodel):
    # State prediction of the tracked objects using camera state estimates
    # INPUTS : TRACK_STATE_ESTIMATES :
    #        : nValidObj :
    #        : motionmodel :
    # OUTPUT : TRACK_STATE_ESTIMATES
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    TRACK_STATE_ESTIMATES = TRACK_STATE_ESTIMATES_in
    if(TRACK_STATE_ESTIMATES.nValidTracks == 0):
        # do not perform state prediction if no objects are present
        return TRACK_STATE_ESTIMATES
