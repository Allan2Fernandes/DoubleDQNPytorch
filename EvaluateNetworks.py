import sys
import time
import numpy as np
import torch
import os
import gymnasium as gym
from Q_Network import Q_Network
from colorama import Fore, Back, Style

models_directory = "C:/Users/Allan/Desktop/Models/LunarLanderModels2"
env = gym.make(
    "LunarLander-v2",
    render_mode='rgb_array',
    continuous= False,
    gravity = -10.0,
    enable_wind= True,
    wind_power= 1.0,
    turbulence_power= 1.5,
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
episodes_to_test = 50
max_time_steps = sys.maxsize

def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32)
    state = torch.unsqueeze(state, dim=0)
    state = state.to(device)
    return state

def get_action(model, state):
    model.eval()
    with torch.no_grad():
        Q_values = int(torch.argmax(model(state)[0]))
        pass
    return Q_values

for model_name in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_name)
    model = Q_Network(device=device, action_space_size=action_size, observation_space_size=state_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model_episode_rewards = []
    for episode in range(episodes_to_test):
        observation, _ = env.reset()
        observation = preprocess_state(observation)
        episode_reward = 0
        for time_step in range(max_time_steps):
            action = get_action(model, observation)
            next_observation, reward, done, truncated, meta_data = env.step(action)
            next_observation = preprocess_state(next_observation)
            episode_reward += reward
            observation = next_observation
            if done or truncated:
                model_episode_rewards.append(episode_reward)
                break
            pass
        pass
    avg_reward = np.mean(model_episode_rewards)
    if avg_reward > 200:
        text_color = Fore.GREEN
    else:
        #os.remove(model_path)
        text_color = Fore.RED
    print(text_color + "Average reward for model, {0} = {1}".format(model_name, avg_reward))



