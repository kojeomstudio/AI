from game.grid_world_template import *

import numpy as np

grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]

#env = GridWorld(grid_size, start, goal, obstacles)
#env.render()

grid_x = 5
grid_y = 5

discount_value = 1.0
action_probability_value = 0.25
action_reward = -1.0

# 랜덤으로 액션을 결정한다.
# 전후좌우 액션중에 하나이므로 각 확률은 25% = 0.25
def get_action_policy():
    return AgentHelper.get_rand_agent_action_type()

def get_state_reward_value():
    return None

def is_can_move(dir : tuple):
    # 0 : x, 1: y
    if dir[0] >= grid_x:
        return False
    elif dir[0] < 0:
        return False
    elif dir[1] >= grid_y:
        return False
    elif dir[1] < 0:
        return False
    return True

def get_reward(dir : tuple):
    return (rl_grid_info[dir[0]][dir[1]] + action_reward) * action_probability_value * discount_value

def calc_rewards(in_x : int, in_y :int):
    up = in_x ,in_y + 1
    down = in_x, in_y - 1
    left = in_x - 1, in_y
    right = in_x + 1, in_y

    origin = in_x, in_y

    reward_sum = rl_grid_info[origin[0]][origin[1]]
    if in_x == (grid_x - 1) and in_y == (grid_y - 1):
        # goal node
        reward_sum = 0
    else:
        if is_can_move(up):
            reward_sum += get_reward(up)
    
        if is_can_move(down):
            reward_sum += get_reward(down)

        if is_can_move(left):
            reward_sum += get_reward(left)

        if is_can_move(right):
            reward_sum += get_reward(right)

    return reward_sum


rl_grid_info = np.zeros((grid_x, grid_y))
#print(f"rl_grid_info \n{rl_grid_info}")

total_episode_num = 500

for episode_idx in range(total_episode_num):
    for x in range(grid_x):
        for y in range(grid_y):
            rl_grid_info[x][y] = calc_rewards(x, y)
    # end episode
    print(rl_grid_info)
    





