import numpy as np


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
    for row in range(FUSION_INFO.shape[0]):
        for col in range(FUSION_INFO.shape[1]):
            FUSION_INFO[row, col].Beta[:] = 0.0
            FUSION_INFO[row, col].BetaSum = 0.0
            FUSION_INFO[row, col].nHypothesis = 0.0

    # NEED TO FILL MORE
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
