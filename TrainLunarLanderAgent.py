import gymnasium as gym
import numpy as np
from DQN import DQN
import torch
from collections import deque

env = gym.make(
    "LunarLander-v2",
    render_mode='human',
    continuous= False,
    gravity = -10.0,
    enable_wind= True,
    wind_power= 1.0,
    turbulence_power= 1.5,
)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')

def expand_state_dims(state):
    #state = torch.unsqueeze(state, dim=-1)
    state = torch.unsqueeze(state, dim=0)
    return state

num_episodes = 100000
num_timesteps = 20000
batch_size = 16

action_size = env.action_space.n
state_size = env.observation_space.shape[0]
dqn = DQN(action_size=action_size, state_size=state_size, device=device)

num_completed_steps = 0
all_episodes_return = deque(maxlen=100)
env.reset()
for episode_number in range(num_episodes):
    total_return = 0
    init_state = env.reset()[0]
    state = init_state
    state = torch.Tensor(state)
    state = expand_state_dims(state)
    #dqn.update_target_network()
    for time_step_number in range(num_timesteps):
        num_completed_steps += 1
        if num_completed_steps % dqn.update_rate == 0:
            dqn.update_target_network()

        action = dqn.epsilon_greedy(state)
        dqn.decay_epsilon()

        next_state, reward, done, truncated, meta_data = env.step(action)
        next_state = torch.tensor(next_state)
        next_state = expand_state_dims(next_state)
        dqn.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_return += reward

        if done or truncated:
            print("Total reward for episode {1}: {0}".format(total_return, episode_number))
            all_episodes_return.append(total_return)
            print("Running average reward = {0}".format(np.mean(np.array(all_episodes_return))))
            break

        if len(dqn.replay_buffer) > batch_size:
            dqn.train_double_DQN(batch_size=batch_size)
            pass

        pass