"""
#################################
# Gain Utility Jain function
#################################
"""

#########################################################
# Function definition


def reward_val(upn2, upn1, ufn2, ufn1, jain, param, uav_r, uav_f, list_up, list_uf, list_sum):
    """
    This function calculates the reward function based on the current and previous throughput values from both the
    primary and the fusion(emergency) networks.
    :param upn2: The current-state throughput value for the primary network.
    :param upn1: The previous-state throughput value for the primary network.
    :param ufn2: The current-state throughput value for the fusion(emergency) network.
    :param ufn1: The previous-state throughput value for the fusion(emergency) network.
    :param jain: The jain value.
    :param param: -
    :param uav_r: -
    :param uav_f: -
    :param list_up: -
    :param list_uf: -
    :param list_sum: The list of summation throughput for both networks.
    :return: This function returns the current reward function, the difference between current and previous throughput
             for each network.
    """
    deltaupn = upn2 - upn1
    deltaufn = ufn2 - ufn1
    reward = 0

    if list_sum.size == 1:
        max_sum = 0
    else:
        max_sum = max(list_sum)

    if ufn2 + upn2 - max_sum > 0:
        reward = 10 * (ufn2 + upn2 - max_sum)
    if ufn2 + upn2 - max_sum == 0:
        reward = 0
    if ufn2 + upn2 - max_sum < 0:
        reward = (ufn2 + upn2 - max_sum)

    return reward, deltaupn, deltaufn
