"""
#################################
# Location Generation function
#################################
"""

#########################################################
# import libraries
from random import seed
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#########################################################
# Function definition


def location(num_UAV, Height, Length, Width, UAV_L_MAX,
             UAV_L_MIN, UAV_W_MAX, UAV_W_MIN, pthDist, SaveFile, Plot, Divider):
    """
    This function returns all random locations of UAVs and ground users based on the constraints.

    :param num_UAV: The number of UAVs
    :param Height: The flight altitude
    :param Length: The length of the grid
    :param Width:  The grid's width
    :param UAV_L_MAX: The maximum length that UAVs can move in the grid size
    :param UAV_L_MIN: The minimum length that UAVs can move in the grid size
    :param UAV_W_MAX: The maximum width that UAVs can move in the grid size
    :param UAV_W_MIN: The minimum width that UAVs can move in the grid size
    :param pthDist: The directory of the location file
    :param SaveFile: A Flag (True, False) for saving the location on a hard dreive
    :param Plot: A Flag (True, False) for plotting the location of Drones
    :param Divider: To scale the grid size based on the requirements
    :return: a dictionary includes all X,Y,Z for all UAVs and terrestrial users
    """

    if SaveFile:
        seed(1)
        X_S = -50/Divider
        Y_S = 50/Divider
        Z_S = 10

        X_F = 150/Divider
        Y_F = 50/Divider
        Z_F = 10

        X_GT = 0/Divider
        Y_GT = 50/Divider
        Z_GT = 0

        X_GR = 100/Divider
        Y_GR = 50/Divider
        Z_GR = 0

        X_U = np.random.randint(UAV_L_MIN, UAV_L_MAX, (num_UAV, 1))
        Y_U = np.random.randint(UAV_W_MIN, UAV_W_MAX, (num_UAV, 1))
        Z_U = np.ones([num_UAV, 1], dtype=int) * int(Height)

        X_Fixed = np.matrix([X_S, X_F, X_GT, X_GR])
        Y_Fixed = np.matrix([Y_S, Y_F, Y_GT, Y_GR])
        Z_Fixed = np.matrix([Z_S, Z_F, Z_GT, Z_GR])

        return_dict = dict([('X_S', X_S), ('Y_S', Y_S), ('Z_S', Z_S), ('X_F', X_F), ('Y_F', Y_F), ('Z_F', Z_F),
                            ('X_GT', X_GT), ('Y_GT', Y_GT), ('Z_GT', Z_GT), ('X_GR', X_GR), ('Y_GR', Y_GR),
                            ('Z_GR', Z_GR), ('X_U', X_U), ('Y_U', Y_U), ('Z_U', Z_U), ('X_Fixed', X_Fixed),
                            ('Y_Fixed', Y_Fixed), ('Z_Fixed', Z_Fixed)])
        sio.savemat(pthDist, return_dict)
    else:
        return_dict = sio.loadmat(pthDist)
        X_U = return_dict.get('X_U')
        X_S = return_dict.get('X_S')
        X_F = return_dict.get('X_F')
        X_GT = return_dict.get('X_GT')
        X_GR = return_dict.get('X_GR')

        Y_U = return_dict.get('Y_U')
        Y_S = return_dict.get('Y_S')
        Y_F = return_dict.get('Y_F')
        Y_GT = return_dict.get('Y_GT')
        Y_GR = return_dict.get('Y_GR')

        Z_U = return_dict.get('Z_U')
        Z_S = return_dict.get('Z_S')
        Z_F = return_dict.get('Z_F')
        Z_GT = return_dict.get('Z_GT')
        Z_GR = return_dict.get('Z_GR')

    if Plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(np.squeeze([X_S, X_F]), np.squeeze([Y_S, Y_F]), np.squeeze([Z_S, Z_F]), 'ro', markersize=12)
        # ax.axis([X_S - 10, 10 + X_F, 0-10, Width + 10, 0 - 10, Height + 10])
        ax.set_xlim(X_S - 10, 10 + X_F)
        ax.set_ylim(0-10, Width + 10)
        ax.set_zlim(0 - 10, Height + 10)
        ax.set_xlabel('X axis (L)')
        ax.set_ylabel('Y axis (W)')
        ax.set_zlabel('Z axis (H)')
        ax.grid(True)
        plt.show(block=False)

        ax.plot(np.squeeze([X_GT, X_GR]), np.squeeze([Y_GT, Y_GR]), np.squeeze([Z_GT, Z_GR]), 'bo', markersize=15)

        ax.plot(X_U[:, 0], Y_U[:, 0], Z_U[:, 0], 'go', markersize=10)
        k = 1
        for i, j, l in zip(X_U[:, 0], Y_U[:, 0], Z_U[:, 0]):
            corr = -0.05  # adds a little correction to put annotation in marker's centrum
            ax.text(i, j, l, '%s' % str(k))
            k += 1

        plt.show(block=False)
    return return_dict
