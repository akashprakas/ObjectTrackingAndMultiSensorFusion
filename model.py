import numpy as np
from numpy import sin, cos


class cvmodel:
    def __init__(self, dT, sigma):
        # creates a 2D nearly constant velocity model
        # INPUT: dT: sampling time
        #        sigma: standard deviation of motion noise
        # OUTPUT:obj.d: object state dimension
        #        obj.F: function handle return a motion transition matrix (4 x 4 matrix)
        #        obj.Q: motion noise covariance (4 x 4 matrix)
        #        obj.f: function handle return state prediction --- (4 x 1 vector)
        # NOTE:  the motion model (and correspondingly the covariance also) assumes that the state vector consists of the order : (px, vx, py, vy)
        # --------------------------------------------------------------------------------------------------------------------------------------------------
        self.type = 'Linear'    # to indicate it is a linear process model
        self.name = 'cv'        # to indicate it is a constant velocity motion model
        self.dim = 4
        self.F = np.array([[1, dT,  0, 0],
                          [0,  1,  0, 0],
                          [0, 0,    1, dT],
                          [0, 0,    0,  1]])

        self.Q = np.array(sigma**2 * [[dT**4/4,  dT**3/2,  0, 0],
                                      [dT**3/2,  dT**2,    0, 0],
                                      [0, 0,     dT**4/4,  dT**3/2],
                                      [0, 0,     dT**3/2,  dT**2]])

    def predict(self, x):
        x_t = self.F.dot(x)
        return x_t


class cvmeasmodelPxPy():
    # creates the measurement model for a 2D nearly constant velocity motion model
    # INPUT: sigmaPx, sigmaPy, sigmaVx, sigmaVy: standard deviation of measurement noise
    # OUTPUT: obj.d: measurement dimension
    #         obj.H: function handle return an observation matrix (4 x 4 matrix)
    #         obj.R: measurement noise covariance (4 x 4 matrix)
    #         obj.h: function handle return a measurement (4 x 1 vector)
    # Its is assumed that the measurements are in the order (px, vx, py, vy)
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.dim = 2
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

    def convertToMeasSpace(self, x):
        x_t = self.h.dot(x)
        return x_t
