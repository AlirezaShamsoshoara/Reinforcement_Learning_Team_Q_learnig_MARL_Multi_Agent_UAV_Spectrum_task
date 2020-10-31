#################################
# CSI function
#################################

#########################################################
# import libraries
import scipy.spatial.distance as ssd
import numpy as np
import scipy.io as sio

#########################################################
# Function definition

###############################
# Load CSI


def load_csi(num_UAV, location, pthH, SaveFile):
    """
    This function generate the CSI parameters based on the LOS propagation model and the location of nodes at the
    beginning of the problem.

    :param num_UAV: Number of UAVs.
    :param location: A dictionary including all location.
    :param pthH: The directory to save the CSI parameters on a file.
    :param SaveFile: A Flag(True, False) to save or load data.
    :return: Returns a Numpy array including CSI parameters.
    """
    if SaveFile:

        X_U = location.get('X_U')
        X_S = location.get('X_S')
        X_F = location.get('X_F')
        X_GT = location.get('X_GT')
        X_GR = location.get('X_GR')

        Y_U = location.get('Y_U')
        Y_S = location.get('Y_S')
        Y_F = location.get('Y_F')
        Y_GT = location.get('Y_GT')
        Y_GR = location.get('Y_GR')

        Z_U = location.get('Z_U')
        Z_S = location.get('Z_S')
        Z_F = location.get('Z_F')
        Z_GT = location.get('Z_GT')
        Z_GR = location.get('Z_GR')

        dist_S_uav = [ssd.euclidean([X_S, Y_S, Z_S], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
        dist_S_uav = np.asarray(dist_S_uav)

        dist_uav_F = [ssd.euclidean([X_F, Y_F, Z_F], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
        dist_uav_F = np.asarray(dist_uav_F)

        dist_GT_uav = [ssd.euclidean([X_GT, Y_GT, Z_GT], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
        dist_GT_uav = np.asarray(dist_GT_uav)

        dist_uav_GR = [ssd.euclidean([X_GR, Y_GR, Z_GR], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
        dist_uav_GR = np.asarray(dist_uav_GR)

        dist_S_uav_Norm = dist_S_uav/min(dist_S_uav)
        dist_uav_F_Norm = dist_uav_F/min(dist_uav_F)
        dist_GT_uav_Norm = dist_GT_uav/min(dist_GT_uav)
        dist_uav_GR_Norm = dist_uav_GR/min(dist_uav_GR)

        h_S_uav = np.multiply(1/(dist_S_uav_Norm**2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
        h_S_uav = h_S_uav.T

        h_uav_F = np.multiply(1/(dist_uav_F_Norm**2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
        h_uav_F = h_uav_F.T

        h_GT_uav = np.multiply(1/(dist_GT_uav_Norm**2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
        h_GT_uav = h_GT_uav.T

        h_uav_GR = np.multiply(1/(dist_uav_GR_Norm**2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
        h_uav_GR = h_uav_GR.T

        csi_h = np.zeros([num_UAV, 4, 1], dtype=complex)

        csi_h[:, 0, :] = h_S_uav
        csi_h[:, 1, :] = h_uav_F
        csi_h[:, 2, :] = h_GT_uav
        csi_h[:, 3, :] = h_uav_GR

        sio.savemat(pthH, {'csi_h': csi_h})
    else:
        csi_h_dict = sio.loadmat(pthH)
        csi_h = csi_h_dict.get('csi_h')
    return csi_h

###############################
# GET CSI


def get_csi(num_UAV, location, x_u, y_u):
    """
    This function updates the CSI location based on the changed location of drones.

    :param num_UAV: Number of UAVs.
    :param location: The initial location of drones and the fixed nodes.
    :param x_u: The updated longitude of UAVs.
    :param y_u: The updated latitude of UAVs.
    :return: It returns an update numpy array for the CSI parameters.
    """
    source_uav = 0
    uav_fusion = 1
    gtuser_uav = 2
    uav_gruser = 3

    X_U = x_u
    X_S = location.get('X_S')
    X_F = location.get('X_F')
    X_GT = location.get('X_GT')
    X_GR = location.get('X_GR')

    Y_U = y_u
    Y_S = location.get('Y_S')
    Y_F = location.get('Y_F')
    Y_GT = location.get('Y_GT')
    Y_GR = location.get('Y_GR')

    Z_U = location.get('Z_U')
    Z_S = location.get('Z_S')
    Z_F = location.get('Z_F')
    Z_GT = location.get('Z_GT')
    Z_GR = location.get('Z_GR')

    dist_S_uav = [ssd.euclidean([X_S, Y_S, Z_S], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
    dist_S_uav = np.asarray(dist_S_uav)

    dist_uav_F = [ssd.euclidean([X_F, Y_F, Z_F], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
    dist_uav_F = np.asarray(dist_uav_F)

    dist_GT_uav = [ssd.euclidean([X_GT, Y_GT, Z_GT], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
    dist_GT_uav = np.asarray(dist_GT_uav)

    dist_uav_GR = [ssd.euclidean([X_GR, Y_GR, Z_GR], [i, j, k]) for i, j, k in zip(X_U, Y_U, Z_U)]
    dist_uav_GR = np.asarray(dist_uav_GR)

    dist_S_uav_Norm = dist_S_uav
    dist_uav_F_Norm = dist_uav_F
    dist_GT_uav_Norm = dist_GT_uav
    dist_uav_GR_Norm = dist_uav_GR

    h_S_uav = np.multiply(1 / (dist_S_uav_Norm ** 2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
    h_S_uav = h_S_uav.T

    h_uav_F = np.multiply(1 / (dist_uav_F_Norm ** 2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
    h_uav_F = h_uav_F.T

    h_GT_uav = np.multiply(1 / (dist_GT_uav_Norm ** 2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
    h_GT_uav = h_GT_uav.T

    h_uav_GR = np.multiply(1 / (dist_uav_GR_Norm ** 2), (np.ones([num_UAV, 1]) + 1j * np.ones([num_UAV, 1])).T)
    h_uav_GR = h_uav_GR.T

    csi_h = np.zeros([num_UAV, 4], dtype=complex)
    csi_h[:, source_uav] = np.squeeze(h_S_uav)
    csi_h[:, uav_fusion] = np.squeeze(h_uav_F)
    csi_h[:, gtuser_uav] = np.squeeze(h_GT_uav)
    csi_h[:, uav_gruser] = np.squeeze(h_uav_GR)
    return csi_h
