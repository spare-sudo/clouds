import numpy as np
import random

# Define the maze
maze = [
    [0, -1, 0, 0, 1],
    [0, -1, 0, -1, -1],
    [0, 0, 0, 0, 0],
    [-1, -1, 0, -1, 0],
    [0, 0, 0, -1, 0]
]

start = (0, 0)  # Starting point
goal = (0, 4)   # Goal point

# Map actions to numbers (0, 1, 2, 3)
# 0 = up, 1 = down, 2 = left, 3 = right
action_dict = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Initialize Q-table with zeros (maze dimensions x number of actions)
q_table = np.zeros((len(maze), len(maze[0]), 4))

alpha = 0.1     # Learning rate
gamma = 0.9     # Discount factor
epsilon = 0.1   # Exploration rate
episodes = 1000 # Number of episodes

def is_valid_position(position):
    row, col = position
    return 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] != -1

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Random action (0, 1, 2, or 3)
    else:
        row, col = state
        return np.argmax(q_table[row, col])  # Exploit the best action (max Q-value)

# Q-learning
for episode in range(episodes):
    state = start
    while state != goal:
        row, col = state
        action = choose_action(state)
        move = action_dict[action]
        next_state = (row + move[0], col + move[1])

        if not is_valid_position(next_state):
            reward = -1  # Penalty for hitting a wall
            next_state = state  # Stay in the same position
        elif next_state == goal:
            reward = 1  # Reward for reaching the goal
        else:
            reward = -0.1  # Small penalty for each move

        # Update Q-value
        next_row, next_col = next_state
        best_next_action = np.max(q_table[next_row, next_col])
        q_table[row, col, action] += alpha * (reward + gamma * best_next_action - q_table[row, col, action])

        # Update state
        state = next_state

    # Decrease exploration rate over time
    epsilon = max(0.01, epsilon * 0.99)

# Print the Q-table
print("Trained Q-Table:")
print(q_table)

# Find the path using the trained Q-table
state = start
path = [state]
while state != goal:
    row, col = state
    action = np.argmax(q_table[row, col])  # Choose the best action based on Q-values
    move = action_dict[action]
    next_state = (row + move[0], col + move[1])
    if not is_valid_position(next_state):
        break
    state = next_state
    path.append(state)

# Print the path taken by the agent
print("Path taken by the agent:", path)
