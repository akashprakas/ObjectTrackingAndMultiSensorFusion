import numpy as np


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
                                                          [posCovIdx[0], posCovIdx[1], posCovIdx[0], posCovIdx[1]]] = CLUSTERS_MEAS.ClusterCovariance[:, :, MeasClstID].reshape(4,)
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
    TRACK_DATA = TRACK_DATA_in
    # % if no unassociated clusters are present then do not set new track
    if(TRACK_DATA_in.nValidTracks == 0):
        return
    nTracksLost = 0
    nSurvivingTracks = 0
    LostTrackIDs = np.zeros((1, 100))
    for idx in range(TRACK_DATA_in.nValidTracks):
        TRACK_DATA.TrackParam[idx] = TrackParamInit
        # % set the track data if the track is not lost
        if(not TRACK_DATA_in.TrackParam[idx].Status.Lost):
            nSurvivingTracks = nSurvivingTracks + 1
            TRACK_DATA.TrackParam[nSurvivingTracks] = TRACK_DATA_in.TrackParam[idx]
        # % reuse the Track IDs if the track is lost
        # TO DO : NEED TO REMOVE THE COMMENTED OUT LINE
        elif(TRACK_DATA_in.TrackParam[idx].Status.Lost):
            nTracksLost = nTracksLost + 1
            LostTrackIDs[nTracksLost] = TRACK_DATA_in.TrackParam[idx].id
            #TRACK_DATA_in = TRACK_MANAGER.SelectAndReuseLostTrackID[TRACK_DATA_in, idx]

    TRACK_DATA.nValidTracks = TRACK_DATA_in.nValidTracks - nTracksLost
    TRACK_DATA.TrackIDList = TRACK_DATA_in.TrackIDList
    TRACK_DATA.IsTrackIDused = TRACK_DATA_in.IsTrackIDused
    TRACK_DATA.FirstAvailableIDindex = TRACK_DATA_in.FirstAvailableIDindex
    TRACK_DATA.LastAvailableIDindex = TRACK_DATA_in.LastAvailableIDindex
    return TRACK_DATA, LostTrackIDs
