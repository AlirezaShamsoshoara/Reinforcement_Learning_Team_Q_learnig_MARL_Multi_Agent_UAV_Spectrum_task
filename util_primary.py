"""
#################################
# Utility for Primary function
#################################
"""

#########################################################
# import libraries
import numpy as np
import math

#########################################################
# Function definition


def utility(action, csi, general, power):
    """
    This utility function calculates the amplify-and-forward (AF) throughput rate for the UAV (primary coalition)
     receiver based on the chosen action at the current state.
    :param action: The chosen action vector to find number of drones in each coalition
    :param csi: All CSI parameters calculated based on the updated location
    :param general: general: The configuration parameters for general parameters, etc.
    :param power: The configuration parameters for transmit powers (dB), etc.
    :return: This function returns the calculated AF throughput.
    """

    h_GT_uav = csi[:, 2]
    h_uav_GR = csi[:, 3]
    amplifForwardRate_Primary = 0
    if general.get('DF'):  # Here we're calculating the Decode and Forward (DF) Rate for the cooperation
        pass
    else:  # Here we're calculating the Amplify and Forward (AF) Rate for the cooperation
        uav_r = np.sum(action)  # 1 = Relay, 0 = Fusion
        uav_r = int(uav_r)
        uav_f = int(general.get('NUM_UAV') - uav_r)
        fraction = np.zeros([uav_r, 1])

        i = 0
        index_rel = np.where(action > 0)[0]
        while i < uav_r:
            fraction[i] = (power.get('Power_pt') * power.get('Power_UAV_pr') * (abs(h_GT_uav[index_rel[i]]))**2 *
                           (abs(h_uav_GR[index_rel[i]]))**2)/(1 + power.get('Power_pt') * power.get('Power_UAV_pr') *
                                                              (abs(h_GT_uav[index_rel[i]]))**2 *
                                                              (abs(h_uav_GR[index_rel[i]]))**2)
            i += 1
        amplifForwardRate_Primary = math.log(1 + sum(fraction), 2)

    return amplifForwardRate_Primary
