
import numpy as np

prob_of_random_action = 0.25
discount_factor = 0.9
epoch = 1000

# 4x4 grid world.
grid_world = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

print(f"{grid_world.shape[0]}x{grid_world.shape[1]} grid world.")

row = 4
col = 4

for e in range(epoch):
    next_i = 0
    next_j = 0

    for i in range(row):
        for j in range(col):
            action = None
            if np.random.uniform(0, 1) < prob_of_random_action:
                action = np.random.choice(['up', 'down', 'left', 'right'])
            else:
                action = np.argmax([grid_world[i-1, j], grid_world[i+1, j], grid_world[i, j-1], grid_world[i, j+1]])

            print(f"action : {action}")

            if action == 'up':
                next_i = i - 1
                next_j = j
            elif action == 'down':
                next_i = i + 1
                next_j = j
            elif action == 'left':
                next_i = i
                next_j = j - 1
            elif action == 'right':
                next_i = i
                next_j = j + 1
            if next_i < 0 or next_i >= row or next_j < 0 or next_j >= col:
                continue
            grid_world[i, j] = grid_world[i, j] + discount_factor * grid_world[next_i, next_j]

print(grid_world)
