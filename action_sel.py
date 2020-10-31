"""
#################################
# action selection function
#################################
"""

#########################################################
# import libraries
from copy import deepcopy
import random
from statefromloc import getstateloc
import numpy as np

#########################################################
# Function definition


def action_explore(x, y, action_list, size, state_list, index):
    """
    This function chooses a random action for all drones and then it updates the location and task based on chosen
    action.

    :param x: The longitude of UAVs.
    :param y: The latitude of UAVs.
    :param action_list: A list including possible actions.
    :param size: The size of the grid plane.
    :param state_list: This list shows the current state of the agents.
    :param index: This is the UAV's index.
    :return: The function returns the chosen random action for a signle UAV, new location, and updated state list.
    """
    flag_explore = True
    # state_list = deepcopy(state_list_first)
    x_new = x
    y_new = y
    chosen_action = -1
    new_state = -1
    while flag_explore:
        left_action = deepcopy(action_list)
        random.seed()
        if x == 0:
            left_action.remove(2)
        if y == 0:
            left_action.remove(0)
        if x == size - 1:
            left_action.remove(3)
        if y == size - 1:
            left_action.remove(1)
        chosen_action = random.choice(left_action)

        # 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay
        if chosen_action == 0:
            x_new = x
            y_new = y - 1
        elif chosen_action == 1:
            x_new = x
            y_new = y + 1
        elif chosen_action == 2:
            x_new = x - 1
            y_new = y
        elif chosen_action == 3:
            x_new = x + 1
            y_new = y
        else:
            x_new = x
            y_new = y
        new_state = getstateloc(x_new, y_new, size)
        exclude_state = np.delete(state_list, index)
        flag_explore = (new_state in exclude_state)
    np.put(state_list, index, new_state)
    return chosen_action, x_new, y_new, state_list


def action_exploit(x, y, action_list, size, state_list, qval):
    """
    This function used the Q values to find the best and optimal action regarding the optimal Q values. Hence, the
    chosen action is based on the best neighbor Q values. This state is based on the greedy action or the exploitation.

    :param x: The longitude of all UAVs.
    :param y: Tha latitude of all UAVs.
    :param action_list: The list of all available actions: 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay.
    :param size: The grid size.
    :param state_list: The current state of the MDP.
    :param qval: The Q value matrix which holds all state-action values.
    :return: This function returns an array of chosen actions for all UAVs, new UAVs' locations, and the new MDP state.
    """
    x_new = [None] * len(x)
    y_new = [None] * len(y)
    chosen_action_greedy = -1
    new_state_list = []
    left_action = [deepcopy(action_list) for _ in range(len(x))]
    # 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay
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
            left_states.append(qval[state_list[0], state_list[1], action0, action1])
            taken_actions.append([action0, action1])

    flag_greedy = True
    while flag_greedy:
        max_index_qval = int(np.argmax(left_states))
        chosen_action_greedy = taken_actions[max_index_qval]
        # 0: Up, 1: down, 2: Left, 3: Right, 4: Fusion, 5: Relay
        for uav in np.arange(len(x)):
            if chosen_action_greedy[uav] == 0:
                x_new[uav] = x[uav]
                y_new[uav] = y[uav] - 1
            elif chosen_action_greedy[uav] == 1:
                x_new[uav] = x[uav]
                y_new[uav] = y[uav] + 1
            elif chosen_action_greedy[uav] == 2:
                x_new[uav] = x[uav] - 1
                y_new[uav] = y[uav]
            elif chosen_action_greedy[uav] == 3:
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
    return np.array(chosen_action_greedy), np.squeeze(x_new), np.squeeze(y_new), new_state_list
