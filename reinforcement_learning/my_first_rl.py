import numpy as np

prob_of_random_action = 0.25
discount_factor = 1.0
epoch = 1
reward_value = -1.0

row = 4
col = 4

def get_reward_value(i, j, grid):
    if i < 0 or i >= row or j < 0 or j >= col:
        return 0
    return discount_factor * prob_of_random_action *reward_value

# 초기 상태값
grid_world = np.full((row, col), 0)

print(f"{grid_world.shape[0]}x{grid_world.shape[1]} grid world.")

for e in range(epoch):

    for i in range(row):
        for j in range(col):
            if i == 3 and j == 3:
                # 목표 지점은 상태 값 업데이트 필요 없음.
                continue

            four_dir_sum = (
                get_reward_value(i+1, j, grid_world) +
                get_reward_value(i-1, j, grid_world) +
                get_reward_value(i, j+1, grid_world) +
                get_reward_value(i, j-1, grid_world)
            )
            grid_world[i, j] = four_dir_sum
            #print(f"grid_world[{i}, {j}] updated to: {new_grid[i, j]}")

print("Final grid_world")
print(grid_world)
print(f"start (0, 0): {grid_world[0, 0]}")
print(f"end (3, 3): {grid_world[3, 3]}")
