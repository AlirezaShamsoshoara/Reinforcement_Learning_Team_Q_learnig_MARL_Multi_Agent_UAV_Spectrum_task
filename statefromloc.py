"""
#################################
# Get State from location function
#################################
"""

#########################################################
# import libraries

#########################################################
# Function definition


def getstateloc(X_U, Y_U, size):
    """
    This function find the state based on the current location and the grid size.

    :param X_U: Location of all UAVs. (X)
    :param Y_U:  Location of all UAVs. (Y)
    :param size: The grid size.
    :return: This function returns the current state.
    """
    state = Y_U * size + X_U
    return state
