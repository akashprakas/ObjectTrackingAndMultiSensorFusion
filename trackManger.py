import numpy as np
import copy
from dataclasses import dataclass


def ComputeTrackInitScore(TRACK_DATA, idx, dT, alpha1, alpha2):
    # Use counter based logic for computing the score for Track initialization recursively
    # An ungated measurement is initialized as a new track and a initialization score is recursively computed , if it is above
    # a threshold then the track is set as a 'confirmed' track
    # INPUTS  : TRACK_DATA : data structure corresponding to Track Data ( for details refer to the script 'SensorFusion_Script3_LOAD_DATA_STRUCTURE_PARAMETERS.m')
    #         : idx        : a track index for refering to the track 'TRACK_DATA.TrackParam(idx)'
    #         : dT         : sample time
    #         : alpha1     : threshold for track gated counter
    #         : alpha2     : threshold for sum of gated and predicted counter
    # OUTPUTS : TRACK_DATA : Track data structure containing the updated track management data
    #         : TrackInitScore : Computed track initialization score
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    if(TRACK_DATA.TrackParam[idx].Status.Gated):
        TRACK_DATA.TrackParam[idx].Quality.GatedCounter = TRACK_DATA.TrackParam[idx].Quality.GatedCounter + 1
    elif(TRACK_DATA.TrackParam[idx].Status.Predicted):
        TRACK_DATA.TrackParam[idx].Quality.PredictedCounter = TRACK_DATA.TrackParam[idx].Quality.PredictedCounter + 1

    TRACK_DATA.TrackParam[idx].Quality.TrackedTime = TRACK_DATA.TrackParam[idx].Quality.TrackedTime + dT
    Gt = TRACK_DATA.TrackParam[idx].Quality.GatedCounter
    Pt = TRACK_DATA.TrackParam[idx].Quality.PredictedCounter
    St = Gt + Pt
    if(St <= alpha2):
        if(Gt >= alpha1):
            TrackInitScore = 1  # % track is confirmed
        else:
            TrackInitScore = 2  # % track is still new

    elif(St > alpha2):
        TrackInitScore = 3  # % track is Lost

    return TrackInitScore


def ChooseNewTrackID(TRACK_DATA):
    # Select a unique ID for Track initialization
    # INPUTS  : TRACK_DATA : data structure corresponding to Track Data, Track Management data is located here
    # OUTPUTS : TRACK_DATA : Track data structure containing the updated track management data
    #         : NewID : Unique track id which shall be used for track initialization
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    maxTrackID = 100
    minTrackID = 0
    # choose a new id
    NewID = TRACK_DATA.TrackIDList[TRACK_DATA.FirstAvailableIDindex]
    # set it as '0' since it has been used
    TRACK_DATA.TrackIDList[TRACK_DATA.FirstAvailableIDindex] = 0
    # mark the id as 'used'
    TRACK_DATA.IsTrackIDused[TRACK_DATA.FirstAvailableIDindex] = 1
    if(TRACK_DATA.FirstAvailableIDindex == maxTrackID):
        TRACK_DATA.FirstAvailableIDindex = minTrackID
    else:
        TRACK_DATA.FirstAvailableIDindex = TRACK_DATA.FirstAvailableIDindex + 1

    return NewID


def INIT_NEW_TRACK(CLUSTERS_MEAS, UNASSOCIATED_CLUSTERS, cntMeasClst, TRACK_DATA_in, dT):
    # Set New Track parameters from the unassociated clusters, the
    # Clusters are from the measurements of either radar and camera sensors.
    # Track Initialization of the Local Tracks from Radar and Camera Sensors systems
    # INPUTS : CLUSTERS_MEAS         : Measurement clusters , ( for details refer to the script 'SensorFusion_Script3_LOAD_DATA_STRUCTURE_PARAMETERS.m')
    #        : UNASSOCIATED_CLUSTERS : measurement clusters not gated with any existing/new tracks
    #        : cntMeasClst           : number of ungated measurement clusters
    #        : TRACK_DATA_in         : data structure corresponding to Track Data
    #        : dT                    : sampling time
    # OUTPUTS : TRACK_DATA : Updated Track Data
    #         : nObjNew    : number of new tracks
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nObjNew = 0
    TRACK_DATA = TRACK_DATA_in
    if(TRACK_DATA.nValidTracks == 0 and cntMeasClst == 0):
        # % if no unassociated clusters and valid objects are present then do not set new track
        return
    posCovIdx = [0, 3]
    velCovIdx = [1, 4]  # StateCovIndex = [1,2,4,5]
    sigmaSq = 2
    alpha1 = 5
    alpha2 = 8
    objIndex = TRACK_DATA.nValidTracks

    # THERE IS SOMETHING THAT NEEDS TO BE ADDED HERE

    # % if the track is a 'new' track update the track init function
    for idx in range(TRACK_DATA.nValidTracks):
        if(TRACK_DATA.TrackParam[idx].Status.New):
            TrackInitScore = ComputeTrackInitScore(
                TRACK_DATA, idx, dT, alpha1, alpha2)
            if(TrackInitScore == 1):  # set the track as 'confirmed' track
                # the track is no more 'new'
                TRACK_DATA.TrackParam[idx].Status.New = False
                # the track is existing/confirmed
                TRACK_DATA.TrackParam[idx].Status.Existing = True
                # the track is not lost
                TRACK_DATA.TrackParam[idx].Status.Lost = False
                # reset the gated counter
                TRACK_DATA.TrackParam[idx].Quality.GatedCounter = 0
                # reset the predicted counter
                TRACK_DATA.TrackParam[idx].Quality.PredictedCounter = 0
            elif(TrackInitScore == 2):   # keep it as 'new' track
                # the track is still 'new'
                TRACK_DATA.TrackParam[idx].Status.New = True
                # the track is not existing/stll not confirmed
                TRACK_DATA.TrackParam[idx].Status.Existing = False
                # the track is not lost
                TRACK_DATA.TrackParam[idx].Status.Lost = False
            elif(TrackInitScore == 3):  # tag the track status as 'lost' for deletion
                # the track is no more 'new'
                TRACK_DATA.TrackParam[idx].Status.New = False
                # the track is not existing
                TRACK_DATA.TrackParam[idx].Status.Existing = False
                # the track is lost
                TRACK_DATA.TrackParam[idx].Status.Lost = True

    for idx in range(cntMeasClst):
        MeasClstID = int(UNASSOCIATED_CLUSTERS[0, idx])

        index = objIndex + nObjNew
        # choose a new Track ID
        newId = ChooseNewTrackID(TRACK_DATA)
        # assign a new ID to the new Track
        TRACK_DATA.TrackParam[index].id = newId
        # Update the Track Status , sensor catch info , and tracked time
        TRACK_DATA.TrackParam[index].Status.New = True
        TRACK_DATA.TrackParam[index].Status.Existing = False
        TRACK_DATA.TrackParam[index].Status.Lost = False
        TRACK_DATA.TrackParam[index].Status.Gated = True
        TRACK_DATA.TrackParam[index].Quality.TrackedTime = TRACK_DATA.TrackParam[index].Quality.TrackedTime + dT
        TRACK_DATA.TrackParam[index].Quality.GatedCounter = TRACK_DATA.TrackParam[index].Quality.GatedCounter + 1
        # Update Track Estimates
        TRACK_DATA.TrackParam[index].StateEstimate.px = CLUSTERS_MEAS.ClusterCenters[0, MeasClstID]
        TRACK_DATA.TrackParam[index].StateEstimate.py = CLUSTERS_MEAS.ClusterCenters[1, MeasClstID]
        TRACK_DATA.TrackParam[index].StateEstimate.vx = 0.0
        TRACK_DATA.TrackParam[index].StateEstimate.vy = 0.0
        TRACK_DATA.TrackParam[index].StateEstimate.ax = 0.0
        TRACK_DATA.TrackParam[index].StateEstimate.ay = 0.0
        TRACK_DATA.TrackParam[index].StateEstimate.ErrCOV[[posCovIdx[0], posCovIdx[0], posCovIdx[1], posCovIdx[1]],
                                                          [posCovIdx[0], posCovIdx[1], posCovIdx[0], posCovIdx[1]]] = copy.deepcopy(CLUSTERS_MEAS.ClusterCovariance[:, :, MeasClstID].reshape(4,))
        TRACK_DATA.TrackParam[index].StateEstimate.ErrCOV[[velCovIdx[0], velCovIdx[0], velCovIdx[1], velCovIdx[1]],
                                                          [velCovIdx[0], velCovIdx[1], velCovIdx[0], velCovIdx[1]]] = np.array([sigmaSq, 0, 0, sigmaSq]).reshape(4,)
        nObjNew = nObjNew + 1
    TRACK_DATA.nValidTracks = TRACK_DATA.nValidTracks + nObjNew

    return TRACK_DATA, nObjNew


def MAINTAIN_EXISTING_TRACK(TRACK_DATA_in, dT):
    # Maintain the existing Track information
    # INPUTS : TRACK_DATA_in         : data structure corresponding to Track Data
    #        : dT                    : sampling time
    # OUTPUTS : TRACK_DATA : Updated Track Data
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    TRACK_DATA = TRACK_DATA_in
    if(TRACK_DATA.nValidTracks == 0):
        # % if no unassociated clusters are present then do not execute this function
        return TRACK_DATA

    thresholdPredCounter = 60  # delete if the track is not gated for 3 seconds continuously
    for idx in range(TRACK_DATA.nValidTracks):
        if(TRACK_DATA.TrackParam[idx].Status.Existing):
            # % if the track gets gated once reset the predicted counter to 0
            if(TRACK_DATA.TrackParam[idx].Status.Gated):
                TRACK_DATA.TrackParam[idx].Quality.PredictedCounter = 0
            # % else increment the predicted counter
            elif(TRACK_DATA.TrackParam[idx].Status.Predicted):
                TRACK_DATA.TrackParam[idx].Quality.PredictedCounter = TRACK_DATA.TrackParam[idx].Quality.PredictedCounter + 1
            # % if consecutive predicted count is >= threshold then delete
            if(TRACK_DATA.TrackParam[idx].Quality.PredictedCounter >= thresholdPredCounter):
                TRACK_DATA.TrackParam[idx].Status.Lost = True
                TRACK_DATA.TrackParam[idx].Status.Existing = False
            TRACK_DATA.TrackParam[idx].Quality.TrackedTime = TRACK_DATA.TrackParam[idx].Quality.TrackedTime + dT

    return TRACK_DATA


def DELETE_LOST_TRACK(TRACK_DATA_in, TrackParamInit):
    # Delete lost track info and reuse the track ID
    # INPUTS  : TRACK_DATA_in  : data structure corresponding to Track Data
    #         : TrackParamInit : track parameters initialized, ( for details refer to the script 'SensorFusion_Script3_LOAD_DATA_STRUCTURE_PARAMETERS.m')
    # OUTPUTS : TRACK_DATA    : Updated Track Data excluding the Lost Track
    #         : LostTrackIDs  : List of IDs from the lost Track
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # this is an unncessary copy and can be rewritten with good code.
    TRACK_DATA = copy.deepcopy(TRACK_DATA_in)
    # % if no unassociated clusters are present then do not set new track
    if(TRACK_DATA_in.nValidTracks == 0):
        return TRACK_DATA, 0
    nTracksLost = 0
    nSurvivingTracks = 0
    LostTrackIDs = np.zeros((1, 100))
    for idx in range(TRACK_DATA_in.nValidTracks):
        TRACK_DATA.TrackParam[idx] = copy.deepcopy(TrackParamInit)
        # % set the track data if the track is not lost
        if(not TRACK_DATA_in.TrackParam[idx].Status.Lost):
            TRACK_DATA.TrackParam[nSurvivingTracks] = TRACK_DATA_in.TrackParam[idx]
            nSurvivingTracks = nSurvivingTracks + 1
        # % reuse the Track IDs if the track is lost
        # TO DO : NEED TO REMOVE THE COMMENTED OUT LINE
        elif(TRACK_DATA_in.TrackParam[idx].Status.Lost):
            LostTrackIDs[0, nTracksLost] = TRACK_DATA_in.TrackParam[idx].id
            nTracksLost = nTracksLost + 1
            # TRACK_DATA_in = TRACK_MANAGER.SelectAndReuseLostTrackID[TRACK_DATA_in, idx]

    TRACK_DATA.nValidTracks = TRACK_DATA_in.nValidTracks - nTracksLost
    TRACK_DATA.TrackIDList = TRACK_DATA_in.TrackIDList
    TRACK_DATA.IsTrackIDused = TRACK_DATA_in.IsTrackIDused
    TRACK_DATA.FirstAvailableIDindex = TRACK_DATA_in.FirstAvailableIDindex
    TRACK_DATA.LastAvailableIDindex = TRACK_DATA_in.LastAvailableIDindex
    return TRACK_DATA, LostTrackIDs


def SET_NEW_TRACK_INFO(TRACK_DATA_in, FUSED_TRACKS, nNewTracks, dT):
    # % Set new track info (Specifically for TRACK to TRACK fusion)
    # % INPUTS  : TRACK_DATA_in  : data structure corresponding to Track Data
    # %         : FUSED_TRACKS   : New Track info for the fused tracks
    # %         : nNewTracks     : number of new Tracks
    # %         : dT             : sampling time
    # % OUTPUTS : TRACK_DATA    : Updated Track Data excluding the Lost Track
    # % ---------------------------------------------------------------------------
    TRACK_DATA = copy.deepcopy(TRACK_DATA_in)
    # % if no unassociated clusters and valid objects are present then do not set new track
    if(TRACK_DATA.nValidTracks == 0 and nNewTracks == 0):
        return TRACK_DATA

    StateParamIndex = [0, 1, 3, 4]
    nObjNew = 0
    objIndex = TRACK_DATA.nValidTracks
    alpha1 = 5
    alpha2 = 8

    # % if the track is a 'new' track update the track init function
    for idx in range(TRACK_DATA.nValidTracks):
        if(TRACK_DATA.TrackParam[idx].Status.New):
            TrackInitScore, TRACK_DATA = ComputeTrackInitScore(
                TRACK_DATA, idx, dT, alpha1, alpha2)
            if(TrackInitScore == 1):  # % set the track as 'confirmed' track
                # % the track is no more 'new'
                TRACK_DATA.TrackParam[idx].Status.New = False
                # % the track is existing/confirmed
                TRACK_DATA.TrackParam[idx].Status.Existing = True
                # % the track is not lost
                TRACK_DATA.TrackParam[idx].Status.Lost = False
                # % reset the gated counter
                TRACK_DATA.TrackParam[idx].Quality.GatedCounter = 0
                # % reset the predicted counter
                TRACK_DATA.TrackParam[idx].Quality.PredictedCounter = 0
            elif(TrackInitScore == 2):  # % keep it as 'new' track
                # % the track is still 'new'
                TRACK_DATA.TrackParam[idx].Status.New = True
                # % the track is not existing/stll not confirmed
                TRACK_DATA.TrackParam[idx].Status.Existing = False
                # % the track is not lost
                TRACK_DATA.TrackParam[idx].Status.Lost = False
            elif(TrackInitScore == 3):  # % tag the track status as 'lost' for deletion
                # % the track is no more 'new'
                TRACK_DATA.TrackParam[idx].Status.New = False
                # % the track is not existing
                TRACK_DATA.TrackParam[idx].Status.Existing = False
                # % the track is lost
                TRACK_DATA.TrackParam[idx].Status.Lost = True

    for idx in range(nNewTracks):  # % iterate over each of the unassocisted Local Tracks

        index = objIndex + nObjNew
        nObjNew = nObjNew + 1
        # % Choose a new Track ID
        newId = ChooseNewTrackID(TRACK_DATA)
        # % assign a new ID to the new Track
        TRACK_DATA.TrackParam[index].id = newId
        # % Update the Track Status , sensor catch info , and tracked time
        TRACK_DATA.TrackParam[index].SensorSource.RadarCatch = FUSED_TRACKS[idx].RadarCatch
        TRACK_DATA.TrackParam[index].SensorSource.CameraCatch = FUSED_TRACKS[idx].CameraCatch
        TRACK_DATA.TrackParam[index].SensorSource.RadarSource = FUSED_TRACKS[idx].RadarSource
        TRACK_DATA.TrackParam[index].SensorSource.CameraSource = FUSED_TRACKS[idx].CameraCatch
        TRACK_DATA.TrackParam[index].SensorSource.RadarCameraCatch = FUSED_TRACKS[idx].RadarCameraCatch
        TRACK_DATA.TrackParam[index].Status.New = FUSED_TRACKS[idx].New
        TRACK_DATA.TrackParam[index].Status.Existing = FUSED_TRACKS[idx].Existing
        TRACK_DATA.TrackParam[index].Status.Predicted = FUSED_TRACKS[idx].Predicted
        TRACK_DATA.TrackParam[index].Status.Gated = FUSED_TRACKS[idx].Gated
        TRACK_DATA.TrackParam[index].Quality.TrackedTime = TRACK_DATA.TrackParam[index].Quality.TrackedTime + dT
        TRACK_DATA.TrackParam[index].Quality.GatedCounter = TRACK_DATA.TrackParam[index].Quality.GatedCounter + 1
        # % Update Track Estimates
        TRACK_DATA.TrackParam[index].StateEstimate.px = FUSED_TRACKS[idx].Xfus[0, 0]
        TRACK_DATA.TrackParam[index].StateEstimate.vx = FUSED_TRACKS[idx].Xfus[1, 0]
        TRACK_DATA.TrackParam[index].StateEstimate.py = FUSED_TRACKS[idx].Xfus[2, 0]
        TRACK_DATA.TrackParam[index].StateEstimate.vy = FUSED_TRACKS[idx].Xfus[3, 0]
        TRACK_DATA.TrackParam[index].StateEstimate.ax = 0
        TRACK_DATA.TrackParam[index].StateEstimate.ay = 0
        TRACK_DATA.TrackParam[index].StateEstimate.ErrCOV[:4,
                                                          :4] = FUSED_TRACKS[idx].Pfus

    TRACK_DATA.nValidTracks = TRACK_DATA.nValidTracks + nObjNew

    return TRACK_DATA


def FORM_NEW_TRACKS_FROM_LOCAL_TRACKS(TRACK_DATA_RAD, TRACK_DATA_CAM, UNGATED_TRACK_INFO):
    # % Group ungated local tracks for determination of new fused track
    # % INPUTS  : TRACK_DATA_RAD     : data structure corresponding to Track Data from radar sensors
    # %         : TRACK_DATA_CAM     : data structure corresponding to Track Data from camera sensors
    # %         : UNGATED_TRACK_INFO : Ungated Local Track info (Camera Local Tracks and Radar Local Tracks)
    # % OUTPUTS : FUSED_TRACKS   : New Track info for the fused tracks
    # %         : nNewTracks     : number of new Tracks
    # % ------------------------------------------------------------------------------------------------------------------------
    nNewTracks = 0
    # % Initialize data structure for New Merged Tracks (Currently these parameters are updated, the remaining parameters shall be updated later)
    dim = 4
    nRadars = 6
    nCameras = 8
    nLocalTracks = 100
    nFusedTracks = 100

    UnGatedRadTrackIdx = np.where(
        UNGATED_TRACK_INFO.UngatedRadarTracks[0, :TRACK_DATA_RAD.nValidTracks])[0]
    UnGatedCamTrackIdx = np.where(
        UNGATED_TRACK_INFO.UngatedCameraTracks[0, :TRACK_DATA_CAM.nValidTracks])[0]
    nUngatedTracksRAD = len(UnGatedRadTrackIdx)
    nUngatedTracksCAM = len(UnGatedCamTrackIdx)

    @dataclass
    class CFUSED_TRACKS:
        # % Track kinematics
        Xfus = np.zeros((dim, 1))  # % px, vx, py, vy of the fused track
        # % noise covariance of the estimated fused track
        Pfus = np.zeros((dim, dim))
        Xrad = np.zeros((dim, 1))  # % px, vx, py, vy of the radar track
        Prad = np.zeros((dim, dim))  # % noise covariance of the radar track
        Xcam = np.zeros((dim, 1))  # % px, vx, py, vy of the camera track
        Pcam = np.zeros((dim, dim))  # % noise covariance of the camera track
        # % Sensor catches
        CameraCatch = False  # % is the track estimated from the camera measurements
        RadarCatch = False  # % is the track estimated from the radar measurements
        RadarCameraCatch = False  # % is the track estimated from Radar & Camera measurements
        # % camera sensors that detected the fused track
        CameraSource = [0 for i in range(nCameras)]
        # % radar sensors that detected the fused track
        RadarSource = [0 for i in range(nRadars)]
        # % Track Status Parameters
        # % is the fused track new (it is new if all the associated local tracks are new)
        New = False
        Existing = False  # % it is existing if at least one associated local track is 'existing'
        Predicted = False  # % it is predicted if all all the associated local tracks are predicted
        Gated = False  # % it is gated if atleast one local track is 'gated'
        # FUSED_TRACKS = FUSED_TRACKS(ones(1, nFusedTracks));

    FUSED_TRACKS = [CFUSED_TRACKS() for _ in range(nFusedTracks)]

    # % if the number of local tracks is '0', then do not execute this function
    if((TRACK_DATA_RAD.nValidTracks == 0) and (TRACK_DATA_CAM.nValidTracks == 0)):

        return FUSED_TRACKS, nNewTracks

    if((nUngatedTracksRAD == 0) and (nUngatedTracksCAM == 0)):
        return FUSED_TRACKS, nNewTracks

    # % initialization of data structures for algorithm execution
    nNewTracks = 0
    # putting negative one so that zero will be a valid id.
    CameraTrackIDs = np.zeros((1, nLocalTracks), dtype=int) + -1
    RadarTrackIDs = np.zeros((1, nLocalTracks), dtype=int)+-1
    isCameraTrackGrouped = [0 for _ in range(
        nLocalTracks)]
    isRadarTrackGrouped = [0 for _ in range(
        nLocalTracks)]
    X_i = np.zeros((dim, 1))
    X_j = np.zeros((dim, 1))
    Xfus = np.zeros((dim, 1))
    Xrad = np.zeros((dim, 1))
    Xcam = np.zeros((dim, 1))
    Pfus = np.zeros((dim, dim))
    Prad = np.zeros((dim, dim))
    Pcam = np.zeros((dim, dim))
    Pspread = np.zeros((dim, dim))
    StateParamIndex = [0, 1, 3, 4]
    posCovIdx = [0, 3]
    velCovIdx = [1, 4]
    gammaPos = 10
    gammaVel = 10

    # % Start the grouping
    for ii in range(nUngatedTracksCAM):  # % loop over only the ungated tracks
        nCamTracks = 0  # % Used later for grouping/merging
        i = UnGatedCamTrackIdx[ii]
        if(not isCameraTrackGrouped[i]):
            isCameraTrackGrouped[i] = True
            # % Update the Camera Track ID here (Used later for grouping)
            CameraTrackIDs[0, nCamTracks] = i
            nCamTracks = nCamTracks + 1
            # % Track State from Camera Track 'i'
            X_i[0, 0] = TRACK_DATA_CAM.TrackParam[i].StateEstimate.px
            X_i[1, 0] = TRACK_DATA_CAM.TrackParam[i].StateEstimate.vx
            X_i[2, 0] = TRACK_DATA_CAM.TrackParam[i].StateEstimate.py
            X_i[3, 0] = TRACK_DATA_CAM.TrackParam[i].StateEstimate.vy
            # %P_i     = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV(StateParamIndex,StateParamIndex);

            # CHECK AND SEE IF THE COVS ARE COMING CORRECTLY
            P_ = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV
            row1 = P_[[StateParamIndex[0]], posCovIdx]
            row2 = P_[[StateParamIndex[1]], velCovIdx]
            row3 = P_[[StateParamIndex[2]], posCovIdx]
            row4 = P_[[StateParamIndex[3]], velCovIdx]

            P_i_pos = np.array([row1, row3])
            P_i_vel = np.array([row2, row4])
            # P_i_pos = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV[:2, :2]
            # P_i_vel = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV[:2, :2]
            # % Find all radar Tracks 'j' which ar[ ]ated with the camera track 'i'
            nRadTracks = 0
            for jj in range(nUngatedTracksRAD):

                j = UnGatedRadTrackIdx[jj]
                if (not isRadarTrackGrouped[j]):
                    # % Track State from Radar Track 'j'
                    X_j[0, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.px
                    X_j[1, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.vx
                    X_j[2, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.py
                    X_j[3, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.vy

                    P_ = TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV
                    row1 = P_[[StateParamIndex[0]], posCovIdx]
                    row2 = P_[[StateParamIndex[1]], velCovIdx]
                    row3 = P_[[StateParamIndex[2]], posCovIdx]
                    row4 = P_[[StateParamIndex[3]], velCovIdx]

                    P_j_pos = np.array([row1, row3])
                    P_j_vel = np.array([row2, row4])

                    # % compute the statistical distance between the Radar Track j and Camera Track i
                    Xpos = X_i[[0, 2], 0] - X_j[[0, 2], 0]
                    Xpos = Xpos.reshape(2, 1)
                    Ppos = P_i_pos + P_j_pos
                    Xvel = X_i[[1, 3], 0] - X_j[[1, 3], 0]
                    Xvel = Xvel.reshape(2, 1)
                    Pvel = P_i_vel + P_j_vel
                    # %dist = X' * (P\X); % Statistical dist
                    distPos = Xpos.transpose().dot(np.linalg.inv(
                        Ppos).dot(Xpos))   # Xpos' * (Ppos\Xpos);
                    distPos = Xpos.transpose().dot(np.linalg.solve(Ppos, Xpos))

                    distVel = Xvel.transpose().dot(np.linalg.inv(Pvel).dot(Xvel))  # * (Pvel\Xvel);
                    distVel = Xvel.transpose().dot(np.linalg.solve(Pvel, Xvel))
                    if(abs(distPos) <= gammaPos and abs(distVel) <= gammaVel):
                        isRadarTrackGrouped[j] = True
                        # % Update the Radar Track ID here (Used later for grouping)
                        RadarTrackIDs[0, nRadTracks] = j
                        nRadTracks = nRadTracks + 1

                # % Find all camera Tracks 'j' which are gated with the camera track 'i'
            for jj in range((ii+1), nUngatedTracksCAM):

                j = UnGatedCamTrackIdx[0, jj]
                if(~isCameraTrackGrouped(j)):
                    # % Track State from Camera Track 'j'
                    X_j[0, 0] = TRACK_DATA_CAM.TrackParam[j].StateEstimate.px
                    X_j[1, 0] = TRACK_DATA_CAM.TrackParam[j].StateEstimate.vx
                    X_j[2, 0] = TRACK_DATA_CAM.TrackParam[j].StateEstimate.py
                    X_j[3, 0] = TRACK_DATA_CAM.TrackParam[j].StateEstimate.vy

                    P_ = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV
                    row1 = P_[[StateParamIndex[0]], posCovIdx]
                    row2 = P_[[StateParamIndex[1]], velCovIdx]
                    row3 = P_[[StateParamIndex[2]], posCovIdx]
                    row4 = P_[[StateParamIndex[3]], velCovIdx]

                    P_j_pos = np.array([row1, row3])
                    P_j_pos = np.array([row2, row4])
                    # P_j_pos = TRACK_DATA_CAM.TrackParam[j].StateEstimate.ErrCOV[posCovIdx, posCovIdx]
                    # P_j_vel = TRACK_DATA_CAM.TrackParam[j].StateEstimate.ErrCOV[velCovIdx, velCovIdx]
                    # % compute the statistical distance between the Radar Track j and Camera Track i

                    Xpos = X_i[[0, 2], 0] - X_j[[0, 2], 0]
                    Xpos = Xpos.reshape(2, 1)
                    Ppos = P_i_pos + P_j_pos
                    Xvel = X_i[[1, 3], 0] - X_j[[1, 3], 0]
                    Xvel = Xvel.reshape(2, 1)
                    Pvel = P_i_vel + P_j_vel

                    distPos = Xpos.transpose().dot(np.linalg.inv(
                        Ppos).dot(Xpos))  # Xpos' * (Ppos\Xpos);
                    distVel = Xvel.transpose().dot(np.linalg.inv(
                        Pvel).dot(Xvel))  # Xvel' * (Pvel\Xvel);

                    distPos = Xpos.transpose().dot(np.linalg.solve(Ppos, Xpos))
                    distVel = Xvel.transpose().dot(np.linalg.solve(Pvel, Xvel))
                    if (distPos <= gammaPos and distVel <= gammaVel):
                        isCameraTrackGrouped[j] = True
                        # % Update the Camera Track ID here (Used later for grouping)
                        CameraTrackIDs[0, nCamTracks] = j
                        nCamTracks = nCamTracks + 1

            # % Compute the Track Cluster estimates (This track has either a camera only cluster of both Radar and Camera)
            Xfus[:] = 0.0
            Xrad[:] = 0.0
            Xcam[:] = 0.0
            Pfus[:] = 0.0
            Prad[:] = 0.0
            Pcam[:] = 0.0
            Pspread[:] = 0.0
            NewTrack = True
            ExistingTrack = False
            GatedTrack = False
            PredictedTrack = True
            RadarCatch = False
            CameraCatch = False
            CameraSource = False
            RadarSource = False
            nLocalTracks = nCamTracks + nRadTracks
            weight = 1/nLocalTracks
            # % weighted mean and covariance from camera tracks
            for idx in range(nCamTracks):
                index = CameraTrackIDs[0, idx]
                X_i[0, 0] = TRACK_DATA_CAM.TrackParam[index].StateEstimate.px
                X_i[1, 0] = TRACK_DATA_CAM.TrackParam[index].StateEstimate.vx
                X_i[2, 0] = TRACK_DATA_CAM.TrackParam[index].StateEstimate.py
                X_i[3, 0] = TRACK_DATA_CAM.TrackParam[index].StateEstimate.vy

                P_ = TRACK_DATA_CAM.TrackParam[i].StateEstimate.ErrCOV
                row1 = P_[[StateParamIndex[0]], StateParamIndex]
                row2 = P_[[StateParamIndex[1]], StateParamIndex]
                row3 = P_[[StateParamIndex[2]], StateParamIndex]
                row4 = P_[[StateParamIndex[3]], StateParamIndex]

                P_i = np.array([row1, row2, row3, row4])
                # P_i = TRACK_DATA_CAM.TrackParam[index].StateEstimate.ErrCOV[:4, :4]
                Xfus = Xfus + weight * X_i
                Pfus = Pfus + weight * P_i
                Xcam = copy.deepcopy(X_i)
                Pcam = copy.deepcopy(P_i)
                CameraCatch = (
                    CameraCatch or TRACK_DATA_CAM.TrackParam[index].SensorSource.CameraCatch)
                CameraSource = (
                    CameraSource or TRACK_DATA_CAM.TrackParam[index].SensorSource.CameraSource)
                NewTrack = (
                    NewTrack and TRACK_DATA_CAM.TrackParam[index].Status.New)
                ExistingTrack = (
                    ExistingTrack or TRACK_DATA_CAM.TrackParam[index].Status.Existing)
                GatedTrack = (
                    GatedTrack or TRACK_DATA_CAM.TrackParam[index].Status.Gated)
                PredictedTrack = (
                    PredictedTrack and TRACK_DATA_CAM.TrackParam[index].Status.Predicted)

            # % weighted mean and covariance from radar tracks
            for idx in range(nRadTracks):
                index = RadarTrackIDs[0, idx]
                X_i[0, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.px
                X_i[1, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.vx
                X_i[2, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.py
                X_i[3, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.vy
                P_ =        TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV
                row1 = P_[[StateParamIndex[0]], StateParamIndex]
                row2 = P_[[StateParamIndex[1]], StateParamIndex]
                row3 = P_[[StateParamIndex[2]], StateParamIndex]
                row4 = P_[[StateParamIndex[3]], StateParamIndex]

                P_i = np.array([row1, row2, row3, row4])
                # P_i = TRACK_DATA_RAD.TrackParam[index].StateEstimate.ErrCOV[:4, :4]
                Xfus = Xfus + weight * X_i
                Pfus = Pfus + weight * P_i
                Xrad = copy.deepcopy(X_i)
                Prad = copy.deepcopy(P_i)
                RadarCatch = (RadarCatch or TRACK_DATA_RAD.TrackParam[
                    index].SensorSource.RadarCatch)
                RadarSource = (RadarSource or TRACK_DATA_RAD.TrackParam[
                    index].SensorSource.RadarSource)
                NewTrack = (
                    NewTrack and TRACK_DATA_RAD.TrackParam[index].Status.New)
                ExistingTrack = (
                    ExistingTrack or TRACK_DATA_RAD.TrackParam[index].Status.Existing)
                GatedTrack = (
                    GatedTrack or TRACK_DATA_RAD.TrackParam[index].Status.Gated)
                PredictedTrack = (
                    PredictedTrack and TRACK_DATA_RAD.TrackParam[index].Status.Predicted)

            NewTrack = not ExistingTrack
            RadarCameraCatch = (RadarCatch and CameraCatch)
            CameraTrackIDs[:] = 0
            RadarTrackIDs[:] = 0
            # % reset to 0
            # I am skipping the pspread that is added to this
            # % update the Merged Track in the output

            FUSED_TRACKS[nNewTracks].Xfus = copy.deepcopy(Xfus)
            FUSED_TRACKS[nNewTracks].Xrad = copy.deepcopy(Xrad)
            FUSED_TRACKS[nNewTracks].Xcam = copy.deepcopy(Xcam)
            FUSED_TRACKS[nNewTracks].Pfus = copy.deepcopy(Pfus)
            FUSED_TRACKS[nNewTracks].Prad = copy.deepcopy(Prad)
            FUSED_TRACKS[nNewTracks].Pcam = copy.deepcopy(Pcam)
            FUSED_TRACKS[nNewTracks].CameraCatch = copy.deepcopy(CameraCatch)
            FUSED_TRACKS[nNewTracks].RadarCatch = copy.deepcopy(RadarCatch)
            FUSED_TRACKS[nNewTracks].RadarCameraCatch = copy.deepcopy(
                RadarCameraCatch)
            FUSED_TRACKS[nNewTracks].CameraSource = copy.deepcopy(CameraSource)
            FUSED_TRACKS[nNewTracks].RadarSource = copy.deepcopy(RadarSource)
            FUSED_TRACKS[nNewTracks].New = copy.deepcopy(NewTrack)
            FUSED_TRACKS[nNewTracks].Existing = copy.deepcopy(ExistingTrack)
            FUSED_TRACKS[nNewTracks].Predicted = copy.deepcopy(PredictedTrack)
            FUSED_TRACKS[nNewTracks].Gated = copy.deepcopy(GatedTrack)
            nNewTracks = nNewTracks + 1

    for ii in range(nUngatedTracksRAD):

        nRadTracks = 0
        i = UnGatedRadTrackIdx[ii]
        if(not isRadarTrackGrouped[i]):
            isRadarTrackGrouped[i] = True
            RadarTrackIDs[0, nRadTracks] = i
            nRadTracks = nRadTracks + 1
            # % Update the Radar Track ID here(Used later for grouping)
            # % Track State from Radar Track 'i'
            X_i[0, 0] = TRACK_DATA_RAD.TrackParam[i].StateEstimate.px
            X_i[1, 0] = TRACK_DATA_RAD.TrackParam[i].StateEstimate.vx
            X_i[2, 0] = TRACK_DATA_RAD.TrackParam[i].StateEstimate.py
            X_i[3, 0] = TRACK_DATA_RAD.TrackParam[i].StateEstimate.vy

            P_ = TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV
            row1 = P_[[StateParamIndex[0]], posCovIdx]
            row2 = P_[[StateParamIndex[1]], velCovIdx]
            row3 = P_[[StateParamIndex[2]], posCovIdx]
            row4 = P_[[StateParamIndex[3]], velCovIdx]

            P_i_pos = np.array([row1, row3])
            P_i_vel = np.array([row2, row3])

            # P_i_pos = TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV[:2, :2]
            # P_i_vel = TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV[:2, :2]
            for jj in range((ii+1), nUngatedTracksRAD):
                j = UnGatedRadTrackIdx[0, jj]
                if(not isRadarTrackGrouped[j]):

                    # % Track State from Radar Track 'j'
                    X_j[0, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.px
                    X_j[1, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.vx
                    X_j[2, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.py
                    X_j[3, 0] = TRACK_DATA_RAD.TrackParam[j].StateEstimate.vy

                    P_j_pos = TRACK_DATA_RAD.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
                    P_j_vel = TRACK_DATA_RAD.TrackParam[j].StateEstimate.ErrCOV[:2, :2]
                    # % compute the statistical distance between the Radar Track j and Camera Track i

                    Xpos = X_i[[0, 2], 0] - X_j[[0, 2], 0]
                    Xpos = Xpos.reshape(2, 1)
                    Ppos = P_i_pos + P_j_pos

                    Xvel = X_i[[1, 3], 0] - X_j[[1, 3], 0]
                    Xvel = Xvel.reshape(2, 1)
                    Pvel = P_i_vel + P_j_vel

                    distPos = Xpos.transpose().dot(np.linalg.inv(
                        Ppos).dot(Xpos))  # Xpos' * (Ppos\Xpos);
                    distVel = Xvel.transpose().dot(np.linalg.inv(
                        Pvel).dot(Xvel))  # Xvel' * (Pvel\Xvel);
                    if(distPos <= gammaPos and distVel <= gammaVel):

                        isRadarTrackGrouped[j] = True

                        # % Update the Radar Track ID here (Used later for grouping)
                        RadarTrackIDs[0, nRadTracks] = j
                        nRadTracks = nRadTracks + 1

            # % Compute the Track Cluster estimates (This track has Radar only cluster)
            Xfus[:] = 0.0
            Xrad[:] = 0.0
            Pfus[:] = 0.0
            Prad[:] = 0.0
            Pcam[:] = 0.0
            Pspread[:] = 0.0
            NewTrack = False
            ExistingTrack = False
            GatedTrack = False
            PredictedTrack = False
            RadarCatch = False
            CameraCatch = False
            CameraSource[:] = False
            RadarSource[:] = False
            nLocalTracks = nRadTracks
            weight = 1/nLocalTracks

            for idx in range(nRadTracks):
                # % weighted mean and covariance from radar tracks
                index = RadarTrackIDs[0, idx]
                X_i[0, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.px
                X_i[1, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.vx
                X_i[2, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.py
                X_i[3, 0] = TRACK_DATA_RAD.TrackParam[index].StateEstimate.vy
                P_ = TRACK_DATA_RAD.TrackParam[i].StateEstimate.ErrCOV
                row1 = P_[[StateParamIndex[0]], StateParamIndex]
                row2 = P_[[StateParamIndex[1]], StateParamIndex]
                row3 = P_[[StateParamIndex[2]], StateParamIndex]
                row4 = P_[[StateParamIndex[3]], StateParamIndex]

                P_i = np.array([row1, row2, row3, row4])
                # P_i = TRACK_DATA_RAD.TrackParam[index].StateEstimate.ErrCOV[:4, :4]
                Xfus = Xfus + weight * X_i
                Pfus = Pfus + weight * P_i
                Xrad = copy.deepcopy(X_i)
                Prad = copy.deepcopy(P_i)
                RadarCatch = (
                    RadarCatch or TRACK_DATA_RAD.TrackParam[index].SensorSource.RadarCatch)
                RadarSource = (
                    np.any(RadarSource) or np.any(TRACK_DATA_RAD.TrackParam[index].SensorSource.RadarSource))
                NewTrack = (
                    NewTrack and TRACK_DATA_RAD.TrackParam[index].Status.New)
                ExistingTrack = (
                    ExistingTrack or TRACK_DATA_RAD.TrackParam[index].Status.Existing)
                GatedTrack = (
                    GatedTrack or TRACK_DATA_RAD.TrackParam[index].Status.Gated)
                PredictedTrack = (
                    PredictedTrack and TRACK_DATA_RAD.TrackParam[index].Status.Predicted)

            NewTrack = not ExistingTrack
            RadarCameraCatch = (RadarCatch and CameraCatch)

            RadarTrackIDs[:] = 0  # % reset to 0
            # % update the Merged Track in the output
            FUSED_TRACKS[nNewTracks].Xfus = copy.deepcopy(Xfus)
            FUSED_TRACKS[nNewTracks].Xrad = copy.deepcopy(Xrad)
            FUSED_TRACKS[nNewTracks].Xcam = copy.deepcopy(Xcam)
            FUSED_TRACKS[nNewTracks].Pfus = copy.deepcopy(Pfus)
            FUSED_TRACKS[nNewTracks].Prad = copy.deepcopy(Prad)
            FUSED_TRACKS[nNewTracks].Pcam = copy.deepcopy(Pcam)
            FUSED_TRACKS[nNewTracks].CameraCatch = copy.deepcopy(CameraCatch)
            FUSED_TRACKS[nNewTracks].RadarCatch = copy.deepcopy(RadarCatch)
            FUSED_TRACKS[nNewTracks].RadarCameraCatch = copy.deepcopy(
                RadarCameraCatch)
            FUSED_TRACKS[nNewTracks].CameraSource = copy.deepcopy(CameraSource)
            FUSED_TRACKS[nNewTracks].RadarSource = copy.deepcopy(RadarSource)
            FUSED_TRACKS[nNewTracks].New = copy.deepcopy(NewTrack)
            FUSED_TRACKS[nNewTracks].Existing = copy.deepcopy(ExistingTrack)
            FUSED_TRACKS[nNewTracks].Predicted = copy.deepcopy(PredictedTrack)
            FUSED_TRACKS[nNewTracks].Gated = copy.deepcopy(GatedTrack)
            nNewTracks = nNewTracks + 1

    return FUSED_TRACKS, nNewTracks
