import gymnasium as gym
import numpy as np
import torch
from DQN import DQN
from collections import deque

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')
num_episodes = 1000
num_timesteps = 20000
batch_size = 32
env = gym.make('CartPole-v1', render_mode='human')
#env.metadata['render_fps'] = 50
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
dqn = DQN(action_size=action_size, state_size=state_size, device=device)
num_completed_steps = 0
all_episodes_return = deque(maxlen=100)

def expand_state_dims(state):
    #state = torch.unsqueeze(state, dim=-1)
    state = torch.unsqueeze(state, dim=0)
    return state

for episode_number in range(1000):
    total_return = 0
    init_state = env.reset()[0]
    state = init_state
    state = torch.Tensor(state)
    state = expand_state_dims(state)
    for time_step in range(num_timesteps):
        num_completed_steps += 1
        if num_completed_steps%dqn.update_rate == 0:
            dqn.update_target_network()

        action = dqn.epsilon_greedy(state)
        dqn.decay_epsilon()
        next_state, reward, done, max_steps, meta_data = env.step(action)
        next_state = torch.tensor(next_state)
        next_state = expand_state_dims(next_state)
        dqn.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_return += reward
        if done or max_steps:
            print("Total reward for episode {1}: {0}".format(total_return, episode_number))
            all_episodes_return.append(total_return)
            print("Running average reward = {0}".format(np.mean(np.array(all_episodes_return))))
            break

        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size=batch_size)
            pass
        pass

