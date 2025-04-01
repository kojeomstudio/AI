import numpy as np

prob_of_random_action = 0.25
prob_of_state = 1.0
discount_factor = 1.0
epoch = 3
reward_value = -1.0

row = 4
col = 4

def get_state_value(i, j, grid):

    grid_value = 1
    if i < 0 or i >= row or j < 0 or j >= col:
        grid_value = 0
    else:
        grid_value = grid[i, j]
    return  prob_of_random_action * (reward_value + (discount_factor * grid_value * prob_of_state))

# 초기 상태값
grid_world = np.full((row, col), 0)

print(f"{grid_world.shape[0]}x{grid_world.shape[1]} grid world.")

for e in range(epoch):

    target_grid = np.copy(grid_world)
    update_grid = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            if i == 3 and j == 3:
                # 목표 지점은 상태 값 업데이트 필요 없음.
                continue

            four_dir_sum = (
                get_state_value(i+1, j, target_grid) +
                get_state_value(i-1, j, target_grid) +
                get_state_value(i, j+1, target_grid) +
                get_state_value(i, j-1, target_grid)
            )
            update_grid[i, j] = four_dir_sum

    grid_world = update_grid
    print(f"Epoch >>> {e} grid_world")
    print(grid_world)

print("Final grid_world")
print(grid_world)
print(f"start (0, 0): {grid_world[0, 0]}")
print(f"end (3, 3): {grid_world[3, 3]}")
