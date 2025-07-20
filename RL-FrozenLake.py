import numpy as np
import gymnasium as gym
import random
import time
env = gym.make('FrozenLake-v1', render_mode = 'ansi')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

for episode in range (num_episodes):
    state = env.reset()[0]
    done = False
    reward_current_episode = 0
    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
           action = np.argmax(q_table[state,:]) 
        else:
           action = env.action_space.sample()
        new_state, reward, done, truncated, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        reward_current_episode += reward
        if done == True:
           break
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    reward_all_episodes = []
    reward_all_episodes.append(reward_current_episode)
reray = np.array(reward_all_episodes)
rewards_per_thousand_episodes = np.array_split(reray,num_episodes/1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
print(q_table)
