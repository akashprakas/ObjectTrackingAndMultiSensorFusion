import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from numpy import sin, cos, arctan2
from sklearn.cluster import DBSCAN


def ctsSnsrToEgoFrame(Zpos, Rpos, Rot, Translation):
    # Perform Coordinate transformation on the sensor data (sensor frame to ego vehicle frame)
    # INPUTS : Zpos : measurement position vector (px, py)
    #          Rpos : measurement position noise covarience matrix (2D)
    #          ROT  : eular Rotation matrix
    #          TRANS: translation vector
    # OUTPUTS : Zcts : coordinate transformed measurements vector (in the order px, py)
    #         : Rcts : rotated measurement noise covariance (in the order px, py)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    #print('Rot ',Rot.shape)
    #print('Zpos ',Zpos.shape)
    #print('Translation ',Translation.shape)
    Zcts = np.dot(Rot, Zpos) + Translation
    #print('Zcts ',Zcts.shape)
    Rcts = np.dot(Rot, np.dot(Rpos, Rot.transpose()))
    #print('Rcts ',Rcts.shape)
    return Zcts, Rcts


def CTS_SENSOR_FRAME_TO_EGO_FRAME(nSensors, nMeas, SENSOR_INSTALLATION_PARAMETERS, MEAS_CAN_BUS, MEAS_TRANSFORMED_ARRAY):
    # Perform coordinate transformation of the sensor data from the sensor frame to ego vehicle frame
    # INPUTS : nSensors : number of sensors
    #          nMeas    : maximum number of measurements
    #          SENSOR_INSTALLATION_PARAMETERS : sensor extrinsic parameters
    #          MEAS_CAN_BUS : array of structure of measurement
    #          MEAS_TRANSFORMED_ARRAY : initialized data structure for coordinate transformed measurements
    # OUTPUT : MEAS_TRANSFORMED_ARRAY : data structure for coordinate transformed measurements with the following fields
    #          MeasArray      : coordinate transformed measurement array  (2D array)
    #          MeasCovariance : coordinate transformed measurement covariance array (3D)
    #          MeasRef        : Measurement reference array (meas ID from array of struct and sensor ID)
    #          ValidMeasCount : array of number of valid  measurements from each sensor
    #          ValidCumulativeMeasCount : cumulative valid measurement number count
    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    MEAS_TRANSFORMED_ARRAY.ValidMeasCount[:] = 0
    TotalCount = 0  # indexin start from zero so that when we have -1+1 we will get zero and index at the zero index
    for snsrIdx in range(nSensors):
        for objIdx in range(nMeas):
            if(MEAS_CAN_BUS[snsrIdx, objIdx].measID != 0):
                # print(snsrIdx,objIdx)
                # extract the measurement parameters
                measVectorPosition = np.array([[MEAS_CAN_BUS[snsrIdx, objIdx].px],  [
                                              MEAS_CAN_BUS[snsrIdx, objIdx].py]])
                #print('measVectorPosition ',measVectorPosition.shape)
                measPosiCovariance = MEAS_CAN_BUS[snsrIdx, objIdx].measNoise
                # Apply the transformation on position , velocity and the measurement covariance
                Z, R = ctsSnsrToEgoFrame(measVectorPosition, measPosiCovariance,
                                         SENSOR_INSTALLATION_PARAMETERS[snsrIdx, 0].RotMat2D, SENSOR_INSTALLATION_PARAMETERS[snsrIdx, 0].TranslationVec)

                # Update the transformed data

                MEAS_TRANSFORMED_ARRAY.ValidMeasCount[0,
                                                      snsrIdx] = MEAS_TRANSFORMED_ARRAY.ValidMeasCount[0, snsrIdx] + 1
                # save the can array index in the measurement array
                MEAS_TRANSFORMED_ARRAY.MeasRef[0, TotalCount] = objIdx

                # sensor ID
                MEAS_TRANSFORMED_ARRAY.MeasRef[1,
                                               TotalCount] = MEAS_CAN_BUS[snsrIdx, objIdx].sensorID
                MEAS_TRANSFORMED_ARRAY.MeasArray[:, TotalCount] = Z.squeeze()
                MEAS_TRANSFORMED_ARRAY.MeasCovariance[:, :, TotalCount] = R
                TotalCount = TotalCount + 1
    MEAS_TRANSFORMED_ARRAY.ValidCumulativeMeasCount = np.cumsum(
        MEAS_TRANSFORMED_ARRAY.ValidMeasCount)

    return MEAS_TRANSFORMED_ARRAY


# TRACK ESTIMATIONS FROM RADAR MEASUREMENTS
#  The following Steps are for Tracks estimation using RADAR Sensors Only (RADAR TRACKS)
#  The steps are as follows :
#  1. Cluster the Concatenated Radar measurements (All radars) by DBSCAN Clustering Algorithm,
#     where each cluster corresponds to a specific object ( Traffic Object like vehicle, pedestrian etc)
#  2. Recompute the Cluster Statistics (Mean and Covariance) WITH RESPECT TO EACH of the radars
#  3. Perform state prediction of the object tracks from time t-1 to t (tracks are detected by th RADAR sensors only)
#  4. Perform Gating of radar clusters from each radars with the predicted tracks
#  5. Perform Data association , radar sensor fusion and state estimation
#  6. Manage Radar Tracks by a Radar Track manager


#finding the mean and var of 
def findMeanAndVarCAM(CAMERA_MEAS_CTS, CAMERA_CLUSTERS, epsPosDBSCAN=5):

    # CAMERAMeasCTS, MANAGE_CLUSTER, CAMERA_CLUSTERS, epsPos
    nCounts = np.sum(CAMERA_MEAS_CTS.ValidMeasCount)
    dataCopy = CAMERA_MEAS_CTS.MeasArray[:, :nCounts].copy()  # 2 x nCounts
    # when we use the dbscan we need things to be n_counts x Features , so transposing this
    datacopyT = dataCopy.transpose()
    dbscan_cluster_model = DBSCAN(
        eps=epsPosDBSCAN, min_samples=1).fit(datacopyT)
    labels = dbscan_cluster_model.labels_
    clusters_found = np.unique(labels).shape[0]
    if -1 in labels:
        clusters_found = clusters_found - 1  # -1 mean not associated measurments
    CAMERA_CLUSTERS.nClusters = clusters_found
    for i in range(clusters_found):
        filter_ = labels == i
        cloud_size = np.sum(filter_)
        points = datacopyT[filter_]
        centroid = np.mean(points, axis=0)
        if (points.shape[0]==1):
            cov = np.eye(2)
        else:
            cov = np.cov(points.transpose())
        CAMERA_CLUSTERS.ClusterCenters[:, i] = centroid
        CAMERA_CLUSTERS.ClusterCovariance[:, :, i] = cov
        CAMERA_CLUSTERS.ClusterIDs[0, i] = i
        CAMERA_CLUSTERS.ClustIDAssig[0, :nCounts][filter_] = i
        CAMERA_CLUSTERS.ClusterSizes[0, i] = cloud_size

    return CAMERA_CLUSTERS, nCounts


def findMeanAndVar(RADAR_MEAS_CTS, RADAR_CLUSTERS, epsPosDBSCAN=5):
    # clusters radar measurements using DBSCAN algorithm
    # INPUT:  RADARMeasCTS : array of radar measurements transformed from sensor frame to ego vehicle frame
    #         MANAGE_CLUSTERS   : A structure of arrays for maintaining information for clustering, contains the following fields
    #         measTypeCore      :
    #         clusterMember     :
    #         clusterMemberFlag :
    #         measurementVisited:
    #         RADAR_CLUSTERS    : initialized structure of radar clusters and related information
    #         epsPos, epsVel    : clustering eps threshold for position and velocity
    # OUTPUT: RADAR_CLUSTERS    : structure of radar clusters and related information with the following fields :
    #         nClusters         : number of clusters formed
    #         ClusterSizes      : number of measurements in each of the clusters
    #         ClusterCenters    : cluster mean(s)
    #         ClusterCovariance : covariance(s) of the cluster mean(s)
    #         ClusterIDs        : cluster ID(s)
    #         ClustIDAssig      : measurement to cluster ID association
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    nCounts = np.sum(RADAR_MEAS_CTS.ValidMeasCount)
    dataCopy = RADAR_MEAS_CTS.MeasArray[:, :nCounts].copy()  # 2 x nCounts
    # when we use the dbscan we need things to be n_counts x Features , so transposing this
    datacopyT = dataCopy.transpose()
    dbscan_cluster_model = DBSCAN(
        eps=epsPosDBSCAN, min_samples=5).fit(datacopyT)
    labels = dbscan_cluster_model.labels_
    clusters_found = np.unique(labels).shape[0]
    if -1 in labels:
        clusters_found = clusters_found - 1  # -1 mean not associated measurments
    RADAR_CLUSTERS.nClusters = clusters_found
    for i in range(clusters_found):
        filter_ = labels == i
        cloud_size = np.sum(filter_)
        points = datacopyT[filter_]
        centroid = np.mean(points, axis=0)
        cov = np.cov(points.transpose())
        RADAR_CLUSTERS.ClusterCenters[:, i] = centroid
        RADAR_CLUSTERS.ClusterCovariance[:, :, i] = cov
        RADAR_CLUSTERS.ClusterIDs[0, i] = i
        RADAR_CLUSTERS.ClustIDAssig[0, :nCounts][filter_] = i
        RADAR_CLUSTERS.ClusterSizes[0, i] = cloud_size

    return RADAR_CLUSTERS, nCounts


def SEGREGATE_CLUSTER(SENSORMeasCTS, SENSOR_CLUSTERS, nCounts):
    # Segregated Clusters, from the combined sensor measurement array and sensor measurement clusters (Output of DBSCAN/NN) ,
    # the cluster centres are recomputed with respect to each of the sensors
    # segregate cluster members (sensor measurements) w.r.t each of the sensor source and recompute the cluster statistics
    # INPUTS : SENSORMeasCTS   : Measurements from Sensors
    #          SENSOR_CLUSTERS : Cluster centers and covariance (Output of DBSCAN/NN)
    # OUTPUT : CLUSTER_DATA    : Segregated Clusters

    # total number of sensors installed around the ego vehicle
    nSensors = len(SENSORMeasCTS.ValidCumulativeMeasCount)
    # total number of measurements received
    nMeas = SENSORMeasCTS.ValidCumulativeMeasCount[-1]
    # total number of measurement clusters (all radars combined)
    nSnsrClsters = SENSOR_CLUSTERS.nClusters
    countTotal = 0

    # Initializations for the output
    @dataclass
    class CCLUSTER_DATA:
        MeasArray = np.zeros((2, int(nSensors)*200), dtype=float)
        MeasCovariance = np.zeros((2, 2, int(nSensors)*200), dtype=float)
        SensorSource = np.zeros((1, int(nSensors)*200), dtype=int)
        ClusterRef = np.zeros((1, int(nSensors)*200), dtype=int)
        ValidMeasCount = np.zeros((1, nSensors), dtype=int)
        ValidCumulativeMeasCount = np.zeros((1, nSensors), dtype=int)

    CLUSTER_DATA = CCLUSTER_DATA()

    for snsrIdx in range(nSensors):
        for idxClstr in range(nSnsrClsters):
            # 1. extract the associated cluster IDs
            clusterID = int(SENSOR_CLUSTERS.ClusterIDs[0, idxClstr])
            # 2. extract the measurement IDs forming the cluster

            measIDsIndex = clusterID == SENSOR_CLUSTERS.ClustIDAssig[:, :nCounts]
            measIDsIndex = np.where(measIDsIndex.squeeze())[0].tolist()
            #import pdb; pdb.set_trace()
            # 3. extract the associated measurement IDs for each of the sensors
            # measIDs = np.where(
            #     snsrIdx == SENSORMeasCTS.MeasRef[1, :nCounts][measIDsIndex])[0]

            measIDs = [
                i for i in measIDsIndex if SENSORMeasCTS.MeasRef[1, i] == snsrIdx]
            if(len(measIDs) > 0):

                weight = 1/len(measIDs)
                x = 0
                R = 0
                for idxMeas in measIDs:
                    x = x + weight * SENSORMeasCTS.MeasArray[:, idxMeas]
                    R = R + weight * \
                        SENSORMeasCTS.MeasCovariance[:, :, idxMeas]

                CLUSTER_DATA.MeasArray[:, countTotal] = x
                CLUSTER_DATA.MeasCovariance[:, :, countTotal] = R
                CLUSTER_DATA.SensorSource[0, countTotal] = snsrIdx
                CLUSTER_DATA.ClusterRef[0, countTotal] = clusterID
                CLUSTER_DATA.ValidMeasCount[0,
                                            snsrIdx] = CLUSTER_DATA.ValidMeasCount[0, snsrIdx] + 1
                countTotal = countTotal + 1
    CLUSTER_DATA.ValidCumulativeMeasCount = np.cumsum(
        CLUSTER_DATA.ValidMeasCount)

    return CLUSTER_DATA
