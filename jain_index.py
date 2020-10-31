"""
#################################
# Jain index function
#################################
"""

#########################################################
# Function definition


def jain(num_f, num_r):
    """
    This function calculates the jain index and the scaled value of that based on the number of UAVs in each
    coalition.

    :param num_f: Number of UAVs in the fusion coalition.
    :param num_r: Number of UAVs in the primary coalition.
    :return: This function returns the jain index (And the Scaled value).
    """
    jain_val = 0.5 * ((num_f + num_r)**2) / (num_f**2 + num_r**2)
    jain_scaled = 4 * jain_val - 3
    return jain_val, jain_scaled
