"""
#################################
# Load the npz file from the local drive and plot the results
#################################
"""

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

#########################################################
# Function definition
Size_list = [2, 3, 4, 5, 6, 8, 10]
Run_list = range(0, 10)
num_Run = 10
Step_list = [120, 800, 2000, 5000, 12000, 20000, 30000]
num_Eps = 200
maximum_sum_util_grid = []
# # ********************************************************************* GRID SIZE = 2 =  2 x 2
sum_utility_size_2 = []
reward_size_2 = []
state_mat_size_2 = []
task_matrix_size_2 = []
xmat_size_2 = []
ymat_size_2 = []
action_size_2 = []
task_diff_size_2 = []
#
for Run in Run_list:
    # outputFile = 'C:\data\Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[0], Run, num_Eps,
    #                                                                                  Step_list[0])
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[0], Run, num_Eps, Step_list[0])
    readfile = np.load(outputFile)
    sum_utility_size_2.append(readfile['sum_utility'])
    # reward_size_2.append(readfile['reward'])
#     task_diff_size_2.append(readfile['task_diff'])
#     # state_mat_size_2.append(readfile['State_Mat'])
#     # task_matrix_size_2.append(readfile['task_matrix'])
#     # xmat_size_2.append(readfile['X_Mat'])
#     # ymat_size_2.append(readfile['Y_Mat'])
#     action_size_2.append(readfile['action'])
#
sum_sum_utility_size_2 = []
# sum_reward_size_2 = []
for Run in Run_list:
    sum_sum_utility_size_2.append(sum(sum_utility_size_2[Run]))
    # sum_reward_size_2.append(sum(reward_size_2[Run]))

maximum_sum_util_2 = max(np.mean(sum_sum_utility_size_2, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_2)
# maximum_sum_reward_2 = max(np.mean(sum_reward_size_2, axis=0))
print ('maximum_sum_util_2 = ', maximum_sum_util_2)
# print ('maximum_sum_reward_2 = ', maximum_sum_reward_2)

# sum_utility_size_2_tr = np.transpose(sum_utility_size_2, (0, 2, 1))
# task_diff_size_2_tr = np.transpose(task_diff_size_2, (0, 2, 1))
# action_size_2_tr = np.transpose(action_size_2, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         sum_goal = []
#         step_goal = Step_list[0]
#         for step in range(0, Step_list[0]):
#             sum_goal.append(sum_utility_size_2_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_2[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         for i, j in action_size_2_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_2, axis=0), markersize='10', linewidth=2.0, color='blue',
#          label="Sum Utility")
# plt.grid(True)
# plt.ylabel('Sum Utility', fontsize=14, fontweight="bold")
# plt.xlabel('Episodes', fontsize=14, fontweight="bold")
# # plt.title('Sum utility 2 x 2')
# plt.legend(prop={'size': 14})
# plt.show(block=False)
# address = '/data/plots/'
# plt.savefig(address + 'SumUtil_Size_%d.pdf' % (Size_list[0]), bbox_inches='tight')


# plt.legend(prop={'size': 14})
# plt.show(block=False)
# address = 'C:\\data\\PDFs\\'
# plt.savefig(address + 'Figure_6_1_all.pdf', bbox_inches='tight')


# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_2, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 2 x 2')
# plt.show(block=False)


# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the Sum utility VS episodes in 2 x 2 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_2_tr, axis=2), axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 2 x 2 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 2 x 2 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_2, sum_sum_utility_size_2, reward_size_2, state_mat_size_2
# del sum_utility_size_2, sum_sum_utility_size_2, reward_size_2, sum_reward_size_2, state_mat_size_2,\
#     sum_utility_size_2_tr, step_goal_list, task_diff_size_2, task_diff_size_2_tr, action_size_2, action_size_2_tr,\
#     movement
# plt.close('all')
# # ********************************************************************* GRID SIZE = 3 =  3 x 3
sum_utility_size_3 = []
reward_size_3 = []
# state_mat_size_3 = []
# task_matrix_size_3 = []
# xmat_size_3 = []
# ymat_size_3 = []
# action_size_3 = []
# task_diff_size_3 = []
#
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[1], Run, num_Eps, Step_list[1])
    readfile = np.load(outputFile)
    sum_utility_size_3.append(readfile['sum_utility'])
    reward_size_3.append(readfile['reward'])
#     task_diff_size_3.append(readfile['task_diff'])
#     # state_mat_size_3.append(readfile['State_Mat'])
#     # task_matrix_size_3.append(readfile['task_matrix'])
#     # xmat_size_3.append(readfile['X_Mat'])
#     # ymat_size_3.append(readfile['Y_Mat'])
#     action_size_3.append(readfile['action'])
#
#
sum_sum_utility_size_3 = []
# sum_reward_size_3 = []
for Run in Run_list:
    sum_sum_utility_size_3.append(sum(sum_utility_size_3[Run]))
    # sum_reward_size_3.append(sum(reward_size_3[Run]))
#     action_size_3.append(readfile['action'])

maximum_sum_util_3 = max(np.mean(sum_sum_utility_size_3, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_3)
# maximum_sum_reward_3 = max(np.mean(sum_reward_size_3, axis=0))
print ('maximum_sum_util_3 = ', maximum_sum_util_3)
# print ('maximum_sum_reward_3 = ', maximum_sum_reward_3)

# sum_utility_size_3_tr = np.transpose(sum_utility_size_3, (0, 2, 1))
# task_diff_size_3_tr = np.transpose(task_diff_size_3, (0, 2, 1))
# action_size_3_tr = np.transpose(action_size_3, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         sum_goal = []
#         step_goal = Step_list[1]
#         for step in range(0, Step_list[1]):
#             sum_goal.append(sum_utility_size_3_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_3[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         for i, j in action_size_3_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_3, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 3 x 3')
# plt.show(block=False)

# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_3, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 3 x 3')
# plt.show(block=False)

# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the goal VS episodes in 3 x 3 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_3_tr, axis=2), axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 3 x 3 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 3 x 3 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_3, sum_sum_utility_size_3, reward_size_3
    # , sum_reward_size_3
    # , state_mat_size_3,\
    # sum_utility_size_3_tr, task_diff_size_3, task_diff_size_3_tr, action_size_3, action_size_3_tr,\
    # movement
plt.close('all')
# # ********************************************************************* GRID SIZE = 4 =  4 x 4
sum_utility_size_4 = []
reward_size_4 = []
state_mat_size_4 = []
task_matrix_size_4 = []
xmat_size_4 = []
ymat_size_4 = []
action_size_4 = []
task_diff_size_4 = []
#
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[2], Run, num_Eps, Step_list[2])
    readfile = np.load(outputFile)
    sum_utility_size_4.append(readfile['sum_utility'])
    # reward_size_4.append(readfile['reward'])
#     task_diff_size_4.append(readfile['task_diff'])
#     # state_mat_size_4.append(readfile['State_Mat'])
#     # task_matrix_size_4.append(readfile['task_matrix'])
#     # xmat_size_4.append(readfile['X_Mat'])
#     # ymat_size_4.append(readfile['Y_Mat'])
#     action_size_4.append(readfile['action'])
#
sum_sum_utility_size_4 = []
# sum_reward_size_4 = []
for Run in Run_list:
    sum_sum_utility_size_4.append(sum(sum_utility_size_4[Run]))
    # sum_reward_size_4.append(sum(reward_size_4[Run]))

maximum_sum_util_4 = max(np.mean(sum_sum_utility_size_4, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_4)
# maximum_sum_reward_4 = max(np.mean(sum_reward_size_4, axis=0))
print ('maximum_sum_util_4 = ', maximum_sum_util_4)
# print ('maximum_sum_reward_4 = ', maximum_sum_reward_4)

# sum_utility_size_4_tr = np.transpose(sum_utility_size_4, (0, 2, 1))
# task_diff_size_4_tr = np.transpose(task_diff_size_4, (0, 2, 1))
# action_size_4_tr = np.transpose(action_size_4, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         sum_goal = []
#         step_goal = Step_list[2]
#         for step in range(0, Step_list[2]):
#             sum_goal.append(sum_utility_size_4_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_4[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         for i, j in action_size_4_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_4, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 4 x 4')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_4, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 4 x 4')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the goal VS episodes in 4 x 4 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_4_tr, axis=2), axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 4 x 4 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 4 x 4 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_4, sum_sum_utility_size_4, reward_size_4
    # , sum_reward_size_4, state_mat_size_4,
    # sum_utility_size_4_tr, step_goal_list, task_diff_size_4, task_diff_size_4_tr, action_size_4, action_size_4_tr,\
    # movement
# plt.close('all')
# # ********************************************************************* GRID SIZE = 5 =  5 x 5
sum_utility_size_5 = []
reward_size_5 = []
state_mat_size_5 = []
task_matrix_size_5 = []
xmat_size_5 = []
ymat_size_5 = []
action_size_5 = []
task_diff_size_5 = []
#
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[3],
                                                                                                          Run, num_Eps,
                                                                                                          Step_list[3])
    readfile = np.load(outputFile)
    sum_utility_size_5.append(readfile['sum_utility'])
#     reward_size_5.append(readfile['reward'])
#     task_diff_size_5.append(readfile['task_diff'])
#     # state_mat_size_5.append(readfile['State_Mat'])
#     # task_matrix_size_5.append(readfile['task_matrix'])
#     # xmat_size_5.append(readfile['X_Mat'])
#     # ymat_size_5.append(readfile['Y_Mat'])
#     action_size_5.append(readfile['action'])
#
sum_sum_utility_size_5 = []
sum_reward_size_5 = []
for Run in Run_list:
    sum_sum_utility_size_5.append(sum(sum_utility_size_5[Run]))
    # sum_reward_size_5.append(sum(reward_size_5[Run]))

maximum_sum_util_5 = max(np.mean(sum_sum_utility_size_5, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_5)
print ('maximum_sum_util_5 = ', maximum_sum_util_5)

# sum_utility_size_5_tr = np.transpose(sum_utility_size_5, (0, 2, 1))
# task_diff_size_5_tr = np.transpose(task_diff_size_5, (0, 2, 1))
# action_size_5_tr = np.transpose(action_size_5, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         sum_goal = []
#         step_goal = Step_list[3]
#         for step in range(0, Step_list[3]):
#             sum_goal.append(sum_utility_size_5_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_5[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         for i, j in action_size_5_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_5, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 5 x 5')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_5, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 5 x 5')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the goal VS episodes in 5 x 5 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_5_tr, axis=2), axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 5 x 5 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 5 x 5 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_5, sum_sum_utility_size_5, reward_size_5
    # sum_reward_size_5, state_mat_size_5,\
    # sum_utility_size_5_tr, task_diff_size_5, task_diff_size_5_tr, action_size_5, action_size_5_tr,\
    # movement
# plt.close('all')
# # ********************************************************************* GRID SIZE = 6 =  6 x 6
sum_utility_size_6 = []
u_fustion_size_6 = []
u_primary_size_6 = []
reward_size_6 = []
state_mat_size_6 = []
task_matrix_size_6 = []
xmat_size_6 = []
ymat_size_6 = []
action_size_6 = []
task_diff_size_6 = []
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[4],
                                                                                                          Run, num_Eps,
                                                                                                          Step_list[4])
    readfile = np.load(outputFile)
    sum_utility_size_6.append(readfile['sum_utility'])
    # u_fustion_size_6.append(readfile['u_fusion'])
    # u_primary_size_6.append(readfile['u_primary'])
    # reward_size_6.append(readfile['reward'])
    # task_diff_size_6.append(readfile['task_diff'])

    # state_mat_size_6.append(readfile['State_Mat'])
    # task_matrix_size_6.append(readfile['task_matrix'])
    # xmat_size_6.append(readfile['X_Mat'])
    # ymat_size_6.append(readfile['Y_Mat'])

    # action_size_6.append(readfile['action'])

sum_sum_utility_size_6 = []
# sum_u_fustion_size_6 = []
# sum_u_primary_size_6 = []
# sum_reward_size_6 = []
for Run in Run_list:
    sum_sum_utility_size_6.append(sum(sum_utility_size_6[Run]))
#     sum_u_primary_size_6.append(sum(u_primary_size_6[Run]))
#     sum_u_fustion_size_6.append(sum(u_fustion_size_6[Run]))
    # sum_reward_size_6.append(sum(reward_size_6[Run]))

maximum_sum_util_6 = max(np.mean(sum_sum_utility_size_6, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_6)
print ('maximum_sum_util_6 = ', maximum_sum_util_6)

# sum_utility_size_6_tr = np.transpose(sum_utility_size_6, (0, 2, 1))
# task_diff_size_6_tr = np.transpose(task_diff_size_6, (0, 2, 1))
# action_size_6_tr = np.transpose(action_size_6, (0, 2, 1, 3))

# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         sum_goal = []
#         step_goal = Step_list[4]
#         for step in range(0, Step_list[4]):
#             sum_goal.append(sum_utility_size_6_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_6[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))

# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         for i, j in action_size_6_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))


# ******************************* SUM PLOT1
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_6, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 6 x 6')
# plt.show(block=False)

# ******************************* SUM PLOT2
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_6, axis=0), markersize='10', linewidth=2.0, color='blue',
#          label="Sum Utility")
# plt.plot(range(0, num_Eps), np.mean(sum_u_primary_size_6, axis=0), markersize='10', linewidth=2.0, color='red',
#          label="Primary Utility")
# plt.plot(range(0, num_Eps), np.mean(sum_u_fustion_size_6, axis=0), markersize='10', linewidth=2.0, color='black',
#          label="Emergency Center Utility")
# plt.grid(True)
# plt.ylabel('Utility(Throughput Rate)', fontsize=14)
# plt.xlabel('Episodes', fontsize=14)
# # plt.xlabel('Episodes', fontsize=14, fontweight="bold")
# plt.legend(prop={'size': 14})
# plt.show(block=False)
# address = 'C:\\data\\PDFs\\'
# plt.savefig(address + 'Figure_6_1_all.pdf', bbox_inches='tight')

# ******************************* SUM PLOT3
# plt.figure()
# d1 = plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_6, axis=0), markersize='10', linewidth=2.0, color='blue',
#             label="Sum Utility", linestyle='-')
# d2 = plt.plot(range(0, num_Eps), np.mean(sum_u_fustion_size_6, axis=0), markersize='10', linewidth=2.0, color='black',
#             label="Emergency Center Utility", linestyle='--')
# plt.grid(True)
# plt.ylabel('Total and Emergency Throughput', fontsize=14)
# plt.xlabel('Episodes', fontsize=14)
# # plt.legend(prop={'size': 14}, loc=10)
# plt2 = plt.twinx()
# d3 = plt.plot(range(0, num_Eps), np.mean(sum_u_primary_size_6, axis=0), markersize='10', linewidth=2.0, color='red',
#           label="Primary Utility", linestyle='-.')
# plt.ylabel('Primary Throughput', fontsize=14, color='red')
# # plt.xlabel('Episodes', fontsize=14, fontweight="bold")
# # plt.legend(prop={'size': 14}, loc=4)
# plt.show(block=False)
# plt_lines = d1 + d2 + d3
# label_text = [line.get_label() for line in plt_lines]
# plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
# address = 'C:\\data\\PDFs\\'
# plt.savefig(address + 'Figure_6_1_all.pdf', bbox_inches='tight')

# ******************************* REWARD PLOT
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_6, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 6 x 6')
# plt.show(block=False)

# ******************************* REQUIRED STEPS PLOT
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', linewidth=2.0, color='black',
#          label="Required Steps")
# plt.grid(True)
# plt.ylabel('Steps', fontsize=14)
# plt.xlabel('Episodes', fontsize=14)
# # plt.title('Required Steps for the goal VS episodes in 6 x 6 Grid')
# plt.legend(prop={'size': 14})
# plt.show(block=False)
# address = 'C:\\data\\PDFs\\'
# plt.savefig(address + 'Figure_6_3_1.pdf', bbox_inches='tight')

# ******************************* SWITCH TASKES PLOT
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_6_tr, axis=2), axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 6 x 6 Grid')
# plt.show(block=False)

# ******************************* MOVEMENT PLOT
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 6 x 6 Grid')
# plt.show(block=False)

# ******************************* SWITCHED TASKS AND MOBILITY PLOT
# plt.figure()
# d1 = plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_6_tr, axis=2), axis=0), linestyle='-', markersize='10',
#          linewidth=2.0, color='red', label='Number of Switched Tasks')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks (-)', fontsize=14, color='red')
# plt.xlabel('Episodes', fontsize=14)
# # plt.legend(prop={'size': 14}, loc=9)
#
# plt2 = plt.twinx()
# d2 = plt.plot(range(0, num_Eps), np.mean(movement, axis=0), linestyle='-.', markersize='10', linewidth=2.0, color='blue',
#           label='Number of movements')
# plt.ylabel('Movement actions (_._)', fontsize=14, color='blue')
# # plt.legend(prop={'size': 14}, loc=10)
# plt.show(block=False)
# plt_lines = d1 + d2
# label_text = [line.get_label() for line in plt_lines]
# plt.legend(plt_lines, label_text, loc=0, prop={'size': 14})
# address = 'C:\\data\\PDFs\\'
# plt.savefig(address + 'Figure_6_7_all.pdf', bbox_inches='tight')

del readfile
del sum_utility_size_6, sum_sum_utility_size_6, reward_size_6
    # sum_reward_size_6, state_mat_size_6,\
    # sum_utility_size_6_tr, task_diff_size_6, task_diff_size_6_tr, action_size_6, action_size_6_tr,\
    # movement
# plt.close('all')
# # ********************************************************************* GRID SIZE = 8 =  8 x 8
sum_utility_size_8 = []
reward_size_8 = []
state_mat_size_8 = []
task_matrix_size_8 = []
xmat_size_8 = []
ymat_size_8 = []
action_size_8 = []
task_diff_size_8 = []
#
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[5],
                                                                                                          Run, num_Eps,
                                                                                                          Step_list[5])
    readfile = np.load(outputFile)
    sum_utility_size_8.append(readfile['sum_utility'])
#     reward_size_8.append(readfile['reward'])
#     task_diff_size_8.append(readfile['task_diff'])
#     # # state_mat_size_8.append(readfile['State_Mat'])
#     # # task_matrix_size_8.append(readfile['task_matrix'])
#     # # xmat_size_8.append(readfile['X_Mat'])
#     # # ymat_size_8.append(readfile['Y_Mat'])
#     action_size_8.append(readfile['action'])
#
# del readfile
sum_sum_utility_size_8 = []
# sum_reward_size_8 = []
for Run in Run_list:
    sum_sum_utility_size_8.append(sum(sum_utility_size_8[Run]))
#     sum_reward_size_8.append(sum(reward_size_8[Run]))

maximum_sum_util_8 = max(np.mean(sum_sum_utility_size_8, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_8)
print ('maximum_sum_util_8 = ', maximum_sum_util_8)

# sum_utility_size_8_tr = np.transpose(sum_utility_size_8, (0, 2, 1))
# task_diff_size_8_tr = np.transpose(task_diff_size_8, (0, 2, 1))
# action_size_8_tr = np.transpose(action_size_8, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         sum_goal = []
#         step_goal = Step_list[5]
#         for step in range(0, Step_list[5]):
#             sum_goal.append(sum_utility_size_8_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_8[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         for i, j in action_size_8_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_8, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 8 x 8')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_8, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 8 x 8')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the goal VS episodes in 8 x 8 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_8_tr, axis=2), axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 8 x 8 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 8 x 8 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_8, sum_sum_utility_size_8, reward_size_8
    # sum_reward_size_8, state_mat_size_8,\
    # sum_utility_size_8_tr, task_diff_size_8, task_diff_size_8_tr, action_size_8, action_size_8_tr,\
    # movement
# plt.close('all')
# # ********************************************************************* GRID SIZE = 10 =  10 x 10
Run_list = range(0, 5)
sum_utility_size_10 = []
reward_size_10 = []
state_mat_size_10 = []
task_matrix_size_10 = []
xmat_size_10 = []
ymat_size_10 = []
action_size_10 = []
task_diff_size_10 = []
for Run in Run_list:
    outputFile = '/data/Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size_list[6],
                                                                                                          Run, num_Eps,
                                                                                                          Step_list[6])
    readfile = np.load(outputFile)
    sum_utility_size_10.append(readfile['sum_utility'])
#     reward_size_10.append(readfile['reward'])
#     task_diff_size_10.append(readfile['task_diff'])
#     # state_mat_size_10.append(readfile['State_Mat'])
#     # task_matrix_size_10.append(readfile['task_matrix'])
#     # xmat_size_10.append(readfile['X_Mat'])
#     # ymat_size_10.append(readfile['Y_Mat'])
#     action_size_10.append(readfile['action'])
#
sum_sum_utility_size_10 = []
# sum_reward_size_10 = []
for Run in Run_list:
    sum_sum_utility_size_10.append(sum(sum_utility_size_10[Run]))
#     sum_reward_size_10.append(sum(reward_size_10[Run]))

maximum_sum_util_10 = max(np.mean(sum_sum_utility_size_10, axis=0))
maximum_sum_util_grid.append(maximum_sum_util_10)
print ('maximum_sum_util_10 = ', maximum_sum_util_10)

# sum_utility_size_10_tr = np.transpose(sum_utility_size_10, (0, 2, 1))
# task_diff_size_10_tr = np.transpose(task_diff_size_10, (0, 2, 1))
# action_size_10_tr = np.transpose(action_size_10, (0, 2, 1, 3))
#
# step_goal_list = [[] for Run in Run_list]
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         sum_goal = []
#         step_goal = Step_list[6]
#         for step in range(0, Step_list[6]):
#             sum_goal.append(sum_utility_size_10_tr[Run][Eps][step])
#             if sum(sum_goal) > 0.99*(sum_sum_utility_size_10[0][1]):
#                 step_goal = step
#                 break
#         # print Eps, " = ", step_goal
#         step_goal_list[Run].append(step_goal)
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))
#
# movement = np.zeros([num_Run, num_Eps], dtype=int)
# for Run in Run_list:
#     for Eps in range(0, num_Eps):
#         timer = time.clock()
#         for i, j in action_size_10_tr[Run][Eps]:
#             if i < 4:
#                 movement[Run, Eps] += 1
#             if j < 4:
#                 movement[Run, Eps] += 1
#         print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_sum_utility_size_10, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Sum Utility')
# plt.xlabel('Episodes')
# # plt.title('Sum utility 10 x 10')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(sum_reward_size_10, axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# # plt.title('Reward 10 x 10')
# plt.show(block=False)
# #
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(step_goal_list, axis=0), markersize='10', color='red')
# plt.grid(True)
# plt.ylabel('Steps')
# plt.xlabel('Episodes')
# # plt.title('Required Steps for the goal VS episodes in 10 x 10 Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(np.sum(task_diff_size_10_tr, axis=2), axis=0), markersize='10', color='black')
# plt.grid(True)
# plt.ylabel('Number of Switched Tasks')
# plt.xlabel('Episodes')
# # plt.title('Number of times for switched tasks VS episodes in 10 x X Grid')
# plt.show(block=False)
#
# plt.figure()
# plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='blue')
# plt.grid(True)
# plt.ylabel('Movement actions')
# plt.xlabel('Episodes')
# # plt.title('Number of movement in each episode in 10 x 10 Grid')
# plt.show(block=False)
#
del readfile
del sum_utility_size_10, sum_sum_utility_size_10, reward_size_10
    # sum_reward_size_10, state_mat_size_10,\
    # sum_utility_size_10_tr, task_diff_size_10, task_diff_size_10_tr, action_size_10, action_size_10_tr,\
    # movement

plt.figure()
plt.plot(Size_list, maximum_sum_util_grid, marker='o', markersize='10', linewidth=2.0, color='blue',
         label="Maximum Utility")
plt.grid(True)
plt.ylabel('Sum Utility', fontsize=14, fontweight="bold")
plt.xlabel('Grid Size', fontsize=14, fontweight="bold")
plt.title('Sum utility - Grid Size')
plt.legend(prop={'size': 14})
plt.show(block=False)
address = '/data/'
plt.savefig(address + 'Max Util_Grid.pdf', bbox_inches='tight')

pass
