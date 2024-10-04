import numpy as np
from enum import Enum
import random

# =========================================================
# Global Constants
GRID_RENDER_VALUE_START = '●'
GRID_RENDER_VALUE_GOAL = '★'
GRID_RENDER_VALUE_OBSTACLE = '■'
GRID_RENDER_VALUE_EMPTY = '□'

AGENT_VALUE = 3

REWARD_GOAL = 10
REWARD_OBSTACLE = -5
REWARD_STEP = -1

# =========================================================

class AgentActionType(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

class AgentHelper:
    @staticmethod
    def get_rand_agent_action_type():
        return random.randint(AgentActionType.UP.value, AgentActionType.LEFT.value)
    

class GridCell:
    """Represents a single cell in the grid world."""
    def __init__(self, position, reward, display_char):
        self.position = position  # (row, col)
        self.reward = reward  # reward value for this cell
        self.display_char = display_char  # Character for rendering

# 그리드월드를 생성하는 클래스. 
class GridWorld:
    def __init__(self, grid_size : tuple, start : tuple, goal : tuple, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.agent_position = start
        self.grid = self.create_grid()

    def create_grid(self):
        """Creates the grid world with cells as GridCell objects."""
        grid = np.empty(self.grid_size, dtype=object)

        # Initialize empty cells
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                grid[row, col] = GridCell((row, col), REWARD_STEP, GRID_RENDER_VALUE_EMPTY)

        # Setting obstacles
        for obstacle in self.obstacles:
            grid[obstacle] = GridCell(obstacle, REWARD_OBSTACLE, GRID_RENDER_VALUE_OBSTACLE)

        # Setting start and goal points
        grid[self.start] = GridCell(self.start, REWARD_STEP, GRID_RENDER_VALUE_START)
        grid[self.goal] = GridCell(self.goal, REWARD_GOAL, GRID_RENDER_VALUE_GOAL)

        return grid

    def reset(self):
        """Resets the agent to the start position."""
        self.agent_position = self.start
        return self.agent_position

    def step(self, action_type : AgentActionType):
        """Takes an action and returns the new state, reward, and done flag."""
        new_position = self.move(self.agent_position, action_type)

        # Check if the new position is out of bounds or hits an obstacle
        if self.is_valid_position(new_position):
            self.agent_position = new_position

        reward = self.get_reward()
        done = self.agent_position == self.goal
        return self.agent_position, reward, done

    def move(self, position, action_type : AgentActionType):
        """Moves the agent based on the action (0: up, 1: right, 2: down, 3: left)."""
        if action_type == AgentActionType.UP:
            return (position[0] - 1, position[1])
        elif action_type == AgentActionType.RIGHT:
            return (position[0], position[1] + 1)
        elif action_type == AgentActionType.DOWN:
            return (position[0] + 1, position[1])
        elif action_type == AgentActionType.LEFT:
            return (position[0], position[1] - 1)

    def is_valid_position(self, position):
        """Check if the position is valid (within bounds and not an obstacle)."""
        # Check if out of bounds
        if not (0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]):
            print("Position out of bounds:", position)
            return False
        
        # Check if obstacle
        if self.grid[position].reward == REWARD_OBSTACLE:
            print("Hit an obstacle at:", position)
            return False
        
        return True

    def get_reward(self):
        """Returns a reward based on the current position."""
        return self.grid[self.agent_position].reward

    def render(self):
        """Prints the grid world with the agent's position."""
        grid_copy = np.array([[cell.display_char for cell in row] for row in self.grid])

        # Place agent on the grid
        grid_copy[self.agent_position] = "A"
        
        print(f"==== grid world rendered ====")
        # Print the grid
        for row in grid_copy:
            print(" ".join(row))
        print("\n")

'''

# Example of usage
grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]

env = GridWorld(grid_size, start, goal, obstacles)
env.render()

# Example of taking steps
actions = [1, 2, 2, 1, 1]  # Right, Down, Down, Right, Right
for action in actions:
    state, reward, done = env.step(action)
    env.render()
    print(f"Reward: {reward}, Done: {done}")
    if done:
        print("Goal reached!")
        break
        
'''