import numpy as np
import matplotlib.pyplot as plt

# Environment size and settings
grid_size = 5
goal_state = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 1)]
actions = ['up', 'down', 'left', 'right']
action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# Initialize Q-table
Q = {}
for x in range(grid_size):
for y in range(grid_size):
Q[(x, y)] = {a: 0 for a in actions}

def get_reward(state):
if state == goal_state:
return 10
elif state in obstacles:
return -10
else:
return -1

def is_valid(state):
x, y = state
return 0 <= x < grid_size and 0 <= y < grid_size and state not in obstacles

def choose_action(state):
if np.random.rand() < epsilon:
return np.random.choice(actions)
else:
return max(Q[state], key=Q[state].get)

# Training
for ep in range(episodes):
state = (0, 0)
while state != goal_state:
action = choose_action(state)
dx, dy = action_map[action]
next_state = (state[0] + dx, state[1] + dy)
if not is_valid(next_state):
next_state = state
reward = get_reward(next_state)
best_next = max(Q[next_state].values())
Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
state = next_state

# Path visualization
state = (0, 0)
path = [state]
while state != goal_state:
action = max(Q[state], key=Q[state].get)
dx, dy = action_map[action]
next_state = (state[0] + dx, state[1] + dy)
if not is_valid(next_state) or next_state == state:
break
path.append(next_state)
state = next_state

# Plot
grid = np.zeros((grid_size, grid_size))
for x, y in obstacles:
grid[x, y] = -1
grid[goal_state] = 10

plt.imshow(grid, cmap='gray_r')
path_x, path_y = zip(*path)
plt.plot(path_y, path_x, marker='o', color='blue')
plt.title("Q-Learning Path from Start to Goal")
plt.gca().invert_yaxis()
plt.savefig("q_learning_path.png")
plt.show()

