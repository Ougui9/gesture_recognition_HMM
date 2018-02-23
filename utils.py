# This file implements correct quaternion averaging.
#
# This method is computationally expensive compared to naive mean averaging.
# If only low accuracy is required (or the quaternions have similar orientations),
# then quaternion averaging can possibly be done through simply averaging the
# components.
#
# Based on:
#
# Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
# "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
# no. 4 (2007): 1193-1197.
# Link: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
#reference: https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py



import numpy
import numpy.matlib as npm
from scipy.sparse.linalg import eigs

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = numpy.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return numpy.real(eigenVectors[:,0].A1)
    # _, vec = eigs(A, k=1)
    #
    # return numpy.real(vec[:,0])