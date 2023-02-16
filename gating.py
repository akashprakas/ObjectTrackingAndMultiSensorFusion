import numpy as np


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
    # GATED_CLSTR_LIST = #CLUSTER_MEASUREMENTS.ClusterRef(1,GATED_CLSTR_INDEX_LIST); #list of gated radar cluster ID
    # GATED_CLSTR_LIST = #unique(GATED_CLSTR_LIST)
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
