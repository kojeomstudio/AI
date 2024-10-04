from game.grid_world_template import *

import numpy as np

grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]

env = GridWorld(grid_size, start, goal, obstacles)
env.render()

# 랜덤으로 액션을 결정한다.
def get_action_policy():
    return AgentHelper.get_rand_agent_action_type()

def get_state_reward_value():
    return None


rl_grid_info = np.zeros((5,5))
print(f"rl_grid_info \n{rl_grid_info}")




