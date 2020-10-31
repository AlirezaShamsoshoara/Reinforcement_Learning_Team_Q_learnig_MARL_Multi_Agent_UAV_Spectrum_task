"""
#################################
# Find Qmax for the Next State
#################################
"""

#########################################################
# import libraries
import numpy as np
from statefromloc import getstateloc
from copy import deepcopy

#########################################################
# Function definition


def find_max_q_next(qval, next_state, x, y, action_list, size):
    """
    This function find the maximum Q values for the future state. Suppose the agents are in state S(t). Then, this
    function finds the best Q values of the future state (S(t+1)).
    :param qval: The Q value matrix from t = 0 until the current time state.
    :param next_state: The future state based on the chosen actions.
    :param x: The current location of UAVs (X).
    :param y: The current location of UAVs (Y).
    :param action_list: The list of available actions: # 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay
    :param size: The grid size.
    :return: This function returns the maximum Q values for the future state to have a consideration for the future
             reward.
    """
    x_new = [None] * len(x)
    y_new = [None] * len(y)
    chosen_action = -1
    new_state_list = []
    left_action = [deepcopy(action_list) for i in range(len(x))]

    for uav in np.arange(len(x)):
        if x[uav] == 0:
            left_action[uav].remove(2)
        if y[uav] == 0:
            left_action[uav].remove(0)
        if x[uav] == size - 1:
            left_action[uav].remove(3)
        if y[uav] == size - 1:
            left_action[uav].remove(1)

    left_states = []
    taken_actions = []
    for action0 in left_action[0]:
        for action1 in left_action[1]:
            left_states.append(qval[next_state[0], next_state[1], action0, action1])
            taken_actions.append([action0, action1])

    maxqval = 0
    flag_greedy = True
    while flag_greedy:
        maxqval = max(left_states)
        max_index_qval = int(np.argmax(left_states))
        chosen_action = taken_actions[max_index_qval]
        # 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay
        for uav in np.arange(len(x)):
            if chosen_action[uav] == 0:
                x_new[uav] = x[uav]
                y_new[uav] = y[uav] - 1
            elif chosen_action[uav] == 1:
                x_new[uav] = x[uav]
                y_new[uav] = y[uav] + 1
            elif chosen_action[uav] == 2:
                x_new[uav] = x[uav] - 1
                y_new[uav] = y[uav]
            elif chosen_action[uav] == 3:
                x_new[uav] = x[uav] + 1
                y_new[uav] = y[uav]
            else:
                x_new[uav] = x[uav]
                y_new[uav] = y[uav]

            state = getstateloc(x_new[uav], y_new[uav], size)
            if state not in new_state_list:
                new_state_list.append(state)
                flag_greedy = False
            else:
                flag_greedy = True
        del left_states[max_index_qval]
        del taken_actions[max_index_qval]

    return maxqval
